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

from lib.Network import  Network
from util.data_val import test_dataset, create_dataloader


def _safe_load_weights(model: torch.nn.Module, ckpt_path: str):
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
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        world_size=world_size,
        rank=local_rank,
    )
    torch.cuda.set_device(local_rank)

    # --- 参数 ---
    parser = argparse.ArgumentParser()
    parser.add_argument('--testsize', type=int, default=512, help='testing size')
    parser.add_argument('--test_batchsize', type=int, default=40,
                        help='testing batch size per GPU')
    parser.add_argument('--pth_path', type=str,
                        default='')
    parser.add_argument('--test_dataset_path', type=str,
                        default='')

    opt = parser.parse_args()
    if opt.test_batchsize < 1:
        raise ValueError('--test_batchsize 必须大于等于 1')

    # 结果累计（按插入顺序保持数据集顺序）
    all_results = {}

    # 根据需求可调整评测数据集顺序
    

    dataset_names = [  'C1', 'Coverage', 'NC16', 'Columbia', 'In-the-Wild','DSO', 'CocoGlide', 'IMD2020', 'Korus' ]
    # dataset_names = ['C1blur3', 'C1blur7', 'C1blur11', 'C1blur15', 'C1blur19', 'C1blur23', 
    #                 'C1jpeg50', 'C1jpeg60', 'C1jpeg70', 'C1jpeg80', 'C1jpeg90', 'C1jpeg100',
    #                 'C1noise3', 'C1noise7', 'C1noise11', 'C1noise15', 'C1noise19', 'C1noise23',
    #                 'CASIA_Facebook', 'CASIA_Wechat', 'CASIA_Weibo', 'CASIA_Whatsapp']

    # --- 构建与加载模型（在 DDP 包裹前加载权重，避免键名不匹配） ---
    model = Network(channels=32).cuda()
    _safe_load_weights(model, opt.pth_path)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    model.eval()

    # --- 逐数据集评测 ---
    for _data_name in dataset_names:
        data_path = os.path.join(opt.test_dataset_path, _data_name)
        save_path = os.path.join(
            './pvt-res',
            f"{os.path.basename(os.path.dirname(opt.pth_path))}-xiao-wuASSSO-4",
            _data_name,
        )
        os.makedirs(save_path, exist_ok=True)

        image_root = os.path.join(data_path, 'Tp')
        gt_root = os.path.join(data_path, 'Gt')
        test_loader = test_dataset(image_root, gt_root, opt.testsize)

        mae_sum = 0.0
        TP_total, FP_total, FN_total = 0.0, 0.0, 0.0

        with torch.no_grad():
            # test_dataset.load_data() 原本一次只读取一张；这里手动组批。
            for batch_start in range(0, test_loader.size, opt.test_batchsize):
                current_batch_size = min(
                    opt.test_batchsize,
                    test_loader.size - batch_start,
                )

                batch_images = []
                batch_gts = []
                batch_names = []

                for _ in range(current_batch_size):
                    image, gt, name, _ = test_loader.load_data()
                    batch_images.append(image)  # 每张为 [1, 3, testsize, testsize]
                    batch_gts.append(np.asarray(gt, np.float32))
                    batch_names.append(name)

                    if local_rank == 0:
                        print(f"> {_data_name} - {name}")

                # 拼成 [B, 3, testsize, testsize]，一次完成前向推理
                images = torch.cat(batch_images, dim=0).cuda(
                    local_rank,
                    non_blocking=True,
                )
                result = model(images)
                batch_logits = result[4]

                # 每张 GT 原始尺寸可能不同，因此逐张恢复到对应 GT 尺寸并统计
                for batch_idx, (gt, name) in enumerate(zip(batch_gts, batch_names)):
                    gt /= (gt.max() + 1e-8)

                    res = F.interpolate(
                        batch_logits[batch_idx:batch_idx + 1],
                        size=gt.shape,
                        mode='bilinear',
                        align_corners=False,
                    )
                    res = torch.sigmoid(res).cpu().numpy().squeeze()

                    # 保留原 sigmoid 概率，用于阈值化统计
                    org = res

                    # 保存可视化（0~255 的 uint8）
                    vis = (
                        (res - res.min())
                        / (res.max() - res.min() + 1e-8)
                        * 255.0
                    ).astype(np.uint8)
                    cv2.imwrite(os.path.join(save_path, name), vis)

                    # 计算 MAE（保持原有计算方式）
                    res_norm = vis.astype(np.float32) / 255.0
                    mae_sum += float(np.mean(np.abs(res_norm - gt)))

                    # 统计 TP / FP / FN
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

        # 仅在 rank 0 上打印该数据集结果
        if local_rank == 0:
            print(
                f"[{_data_name}] F1: {f1:.5f}, IoU: {iou:.5f}, "
                f"MAE: {mae:.5f}, P: {precision:.5f}, R: {recall:.5f}"
            )

        all_results[_data_name] = {
            'F1': f1,
            'IoU': iou,
            'MAE': mae,
            'Precision': precision,
            'Recall': recall,
        }

    # --- 写 CSV 到日志目录（仅 rank 0） ---
    if local_rank == 0:
        log_dir = os.path.dirname(opt.pth_path) if opt.pth_path else '.'
        os.makedirs(log_dir, exist_ok=True)
        csv_file = os.path.join(log_dir, 'Test.csv')

        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            headers = ['Metric'] + list(all_results.keys())
            writer.writerow(headers)

            def row_for(metric):
                return [metric] + [
                    f"{all_results[d][metric]:.5f}" for d in all_results.keys()
                ]

            writer.writerow(row_for('F1'))
            writer.writerow(row_for('IoU'))
            writer.writerow(row_for('MAE'))
            writer.writerow(row_for('Precision'))
            writer.writerow(row_for('Recall'))

            writer.writerow([])
            writer.writerow(['Test Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')])
            writer.writerow(['Model', opt.pth_path])
            writer.writerow(['Test Batch Size Per GPU', opt.test_batchsize])

        print(f"Results saved to {csv_file}")

    # 同步并清理
    dist.barrier()
    dist.destroy_process_group()


if __name__ == '__main__':
    # 指定可见 GPU；若只给一个 GPU，world_size 会等于 1
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '1217'

    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError('没有检测到可用 CUDA GPU')

    # 使用 mp.spawn 以兼容多卡；若仅 1 卡也能正常运行
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)
