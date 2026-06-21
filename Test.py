import os
import csv
from datetime import datetime
import argparse

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP

from lib.Network import Network, Network_pvt, Network_pvt_duo, Network_pvt_duo_2, Network_pvt_duo_3, \
    Network_re50_3Domain,Network_pvt_duo_3_3Domain_gai,Network_res2net50_3Domain # 保留原有导入
from util.data_val import test_dataset, create_dataloader  # 保留原有导入


def _safe_load_weights(model: torch.nn.Module, ckpt_path: str):
    """兼容单卡/多卡保存的权重（是否带有"module."前缀），并在DDP包裹前加载。
    """
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if any(k.startswith("module.") for k in state.keys()):
        state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[load_state_dict] missing={missing}, unexpected={unexpected}")


def main(local_rank: int, world_size: int):
    # --- 初始化分布式环境 ---
    dist.init_process_group(backend="nccl", init_method='env://', world_size=world_size, rank=local_rank)
    torch.cuda.set_device(local_rank)

    # --- 参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=512, help='testing size')
    parser.add_argument('--pth_path', type=str,
                        default='Net_epoch_best.pth')
    # parser.add_argument('--test_dataset_path', type=str, default='../IML-DS/test/')
    parser.add_argument('--test_dataset_path', type=str, default='Vis-final')

    opt = parser.parse_args()

    # 结果累计（按插入顺序保持数据集顺序）
    all_results = {}

    # 根据需求可调整评测数据集顺序
    # dataset_names = ['Columbia', 'Coverage', 'C1',  'NC16', 'DSO', 'In-the-Wild', 'CocoGlide', 'IMD2020',  'Korus','Choice']
    
    dataset_names = ['Choice']
    
    # dataset_names = ['C1blur3', 'C1blur7', 'C1blur11', 'C1blur15', 'C1blur19', 'C1blur23', 
    #                 'C1jpeg50', 'C1jpeg60', 'C1jpeg70', 'C1jpeg80', 'C1jpeg90', 'C1jpeg100',
    #                 'C1noise3', 'C1noise7', 'C1noise11', 'C1noise15', 'C1noise19', 'C1noise23',
    #                 'CASIA_Facebook', 'CASIA_Wechat', 'CASIA_Weibo', 'CASIA_Whatsapp']
    
    # dataset_names = ['CASIA_Facebook', 'CASIA_Wechat', 'CASIA_Weibo', 'CASIA_Whatsapp']
    

    # --- 构建与加载模型（在DDP包裹前加载权重，避免键名不匹配） ---
    model = Network_res2net50_3Domain(channels=32).cuda()
    _safe_load_weights(model, opt.pth_path)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.eval()

    # --- 逐数据集评测 ---
    for _data_name in dataset_names:
        data_path = os.path.join(opt.test_dataset_path, _data_name)
        save_path = os.path.join('./res', f"{os.path.basename(os.path.dirname(opt.pth_path))}-Test-Vis-final", _data_name)
        os.makedirs(save_path, exist_ok=True)

        image_root = os.path.join(data_path, 'Tp')
        gt_root = os.path.join(data_path, 'Gt')
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        mae_sum = 0.0
        TP_total, FP_total, FN_total = 0.0, 0.0, 0.0

        with torch.no_grad():
            for i in range(test_loader.size):
                image, gt, name, _ = test_loader.load_data()
                if local_rank == 0:
                    print(f"> {_data_name} - {name}")

                gt = np.asarray(gt, np.float32)
                gt /= (gt.max() + 1e-8)

                image = image.cuda()
                result = model(image)  # 期望result为list/tuple，取第5个分支用于评测

                # 将分支输出上采样到GT大小
                res = F.interpolate(result[4], size=gt.shape, mode='bilinear', align_corners=False)
                res = res.sigmoid().data.cpu().numpy().squeeze()

                # 用未归一化的sigmoid概率阈值0.5做二值化统计
                org = res  # 这里保留原概率用于阈值化

                # 保存可视化（0~255的uint8）
                vis = ((res - res.min()) / (res.max() - res.min() + 1e-8) * 255.0).astype(np.uint8)
                cv2.imwrite(os.path.join(save_path, name), vis)

                # 计算MAE（在[0,1]范围内）
                res_norm = vis.astype(np.float32) / 255.0
                mae_sum += float(np.mean(np.abs(res_norm - gt)))

                # 统计TP/FP/FN
                pred_bin = (org >= 0.5).astype(np.float32)
                gt_bin = (gt >= 0.5).astype(np.float32)

                TP_total += float((pred_bin * gt_bin).sum())
                FP_total += float((pred_bin * (1 - gt_bin)).sum())
                FN_total += float(((1 - pred_bin) * gt_bin).sum())

        eps = 1e-8
        precision = TP_total / (TP_total + FP_total + eps)
        recall = TP_total / (TP_total + FN_total + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)
        iou = TP_total / (TP_total + FP_total + FN_total + eps)
        mae = mae_sum / test_loader.size

        # 仅在rank0上打印该数据集结果
        if local_rank == 0:
            print(f"[{_data_name}] F1: {f1:.5f}, IoU: {iou:.5f}, MAE: {mae:.5f}, P: {precision:.5f}, R: {recall:.5f}")

        all_results[_data_name] = {
            'F1': f1,
            'IoU': iou,
            'MAE': mae,
            'Precision': precision,
            'Recall': recall
        }

    # --- 写CSV到日志目录（仅rank0） ---
    if local_rank == 0:
        log_dir = os.path.dirname(opt.pth_path) if opt.pth_path else '.'
        os.makedirs(log_dir, exist_ok=True)
        csv_file = os.path.join(log_dir, 'Test-Vis-final.csv')

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Metric'] + list(all_results.keys())
            writer.writerow(headers)

            def row_for(metric):
                return [metric] + [f"{all_results[d][metric]:.5f}" for d in all_results.keys()]

            writer.writerow(row_for('F1'))
            writer.writerow(row_for('IoU'))
            writer.writerow(row_for('MAE'))
            writer.writerow(row_for('Precision'))
            writer.writerow(row_for('Recall'))

            writer.writerow([])
            writer.writerow(['Test Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Model', opt.pth_path])

        print(f"Results saved to {csv_file}")

    # 同步并清理
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # 指定可见GPU；若只给一个GPU，world_size会=1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1216'

    world_size = torch.cuda.device_count()

    # 使用mp.spawn以兼容多卡；若仅1卡也能正常运行
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)



