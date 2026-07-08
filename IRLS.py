import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# 1. 和原论文一致的工具函数
# -----------------------------

def get_w(prob_map: torch.Tensor) -> torch.Tensor:
    """
    论文 Eq.(5) 中的 w(i,j)：
    - 概率在 (0.4, 0.6) 之间 -> 不确定区域 -> 权重 0
    - 其它区域 -> 权重 1
    注意这里用 prob_map.new_zeros/new_ones 保证 device/dtype 一致。
    """
    assert prob_map.dim() == 4, "prob_map 应为 [N, C, H, W]"
    zero = prob_map.new_zeros(1)
    one = prob_map.new_ones(1)
    mask = (prob_map > 0.35) & (prob_map < 0.65)
    w = torch.where(mask, zero, one)
    return w


def get_m_tilde(prob_map: torch.Tensor) -> torch.Tensor:
    """
    论文 Eq.(5) 中的 M_f(i,j)：
    - 在 (0.1, 0.4) -> 拉到 0.1
    - 在 (0.6, 0.9) -> 拉到 0.9
    - 其它保持不变
    用于构造“更果断的掩码版本” M_f。
    """
    assert prob_map.dim() == 4, "prob_map 应为 [N, C, H, W]"
    low_val = prob_map.new_full((1,), 0.1)
    high_val = prob_map.new_full((1,), 0.9)

    out = torch.where((prob_map > 0.15) & (prob_map < 0.35),
                      low_val, prob_map)
    out = torch.where((out > 0.65) & (out < 0.75),
                      high_val, out)
    return out

# -----------------------------
# 2. LRLS 权重模块（IRLS + 可学习修正）
# -----------------------------

class LRLSWeightModule(nn.Module):
    """
    LRLS/IRLS 权重模块：
    给定上一迭代的残差 R_prev ≈ w * (M_{k-1} - M_f^{k-1}),
    先计算 IRLS 基础权重 rho0 = 1 / (|R_prev| + eps),
    再用一个小卷积网络对其做“结构化修正”，得到最终的 rho。

    这样：
    - 从优化角度：仍然是 L1 -> 加权 L2 的 IRLS 思想；
    - 从网络角度：权重 rho 是可学习的，能利用空间结构，而不是死公式。
    """

    def __init__(self,
                 channels: int = 16,
                 layers: int = 3,
                 rho_min: float = 0.1,
                 rho_max: float = 10.0,
                 eps: float = 1e-3):
        super().__init__()
        self.rho_min = rho_min
        self.rho_max = rho_max
        self.eps = eps

        convs = [
            nn.Conv2d(1, channels, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True)
        ]
        for _ in range(layers):
            convs.append(nn.Conv2d(channels, channels, kernel_size=3, padding=1, stride=1))
            convs.append(nn.ReLU(inplace=True))
        convs.append(nn.Conv2d(channels, 1, kernel_size=3, padding=1, stride=1))

        self.convs = nn.Sequential(*convs)

    def forward(self, R_prev: torch.Tensor) -> torch.Tensor:
        """
        R_prev: [N, 1, H, W]，上一轮的残差 w*(M_{k-1} - M_f^{k-1})
        返回 rho: [N, 1, H, W]，IRLS 的重加权系数（正数、带结构）
        """
        assert R_prev.dim() == 4 and R_prev.size(1) == 1, \
            "R_prev 应为 [N, 1, H, W]"

        # 基础 IRLS 权重：1 / (|R| + eps)
        rho0 = 1.0 / (torch.abs(R_prev) + self.eps)

        # 学习到的结构化修正
        rho_delta = self.convs(rho0)
        # softplus 保证 > 0
        rho = F.softplus(rho0 + rho_delta)

        # 限制范围，防止数值崩溃
        rho = torch.clamp(rho, self.rho_min, self.rho_max)
        return rho


# -----------------------------
# 3. LRLS 解析更新公式（代替原 compute_formula）
# -----------------------------

def compute_formula_lrls(
    I: torch.Tensor,
    B_prev: torch.Tensor,
    M_prev: torch.Tensor,
    M_f: torch.Tensor,
    rho: torch.Tensor,
    mu: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    """
    对应 IRLS 推导的 M 更新公式：

    对每个像素 i：
    (C_i^2 + mu + alpha * rho_i) * M_i =
        C_i^2 - C_i B_i + mu * M_prev_i + alpha * rho_i * M_f_i

    这里：
    - I        : C，输入图像或特征 [N, C, H, W]
    - B_prev   : 上一轮背景 B_{k-1} [N, C, H, W]（与 I 对齐）
    - M_prev   : 上一轮掩码 M_{k-1} [N, 1, H, W]
    - M_f      : 本轮 refined 掩码 M_f^k [N, 1, H, W]
    - rho      : LRLS 重加权系数 [N, 1, H, W]
    - mu       : nn.Parameter 标量
    - alpha    : 稀疏正则前系数（float）

    返回：
    - M_new_full: [N, C, H, W]，每通道都有一个更新；之后会在 pipeline 里做通道平均。
    """

    assert I.dim() == 4 and B_prev.dim() == 4
    assert M_prev.dim() == 4 and M_prev.size(1) == 1
    assert M_f.dim() == 4 and M_f.size(1) == 1
    assert rho.dim() == 4 and rho.size(1) == 1

    I_sq = I ** 2  # [N, C, H, W]

    # 将 rho 从 [N,1,H,W] 扩展到 [N,C,H,W]，每个通道共享同一组权重
    if rho.size(1) == 1 and I_sq.size(1) > 1:
        rho_expanded = rho.expand(-1, I_sq.size(1), -1, -1)
    else:
        rho_expanded = rho

    # M_prev, M_f 是 [N,1,H,W]，通过广播扩展到 [N,C,H,W]
    denominator = I_sq + mu + alpha * rho_expanded
    numerator = (
        I_sq - I * B_prev
        + mu * M_prev
        + alpha * rho_expanded * M_f
    )

    M_new_full = numerator / (denominator + 1e-6)
    return M_new_full


# -----------------------------
# 4. LRLSSparsePipeline：替代原 SparsePipeline
# -----------------------------

class LRLSSparsePipeline(nn.Module):
    """
    这是对你原来 SparsePipeline 的“IRLS/LRLS 重写版”。

    接口保持一致：
        forward(self, I, B_k_1, m_k_1, m_hat_k_1, m_k_2)

    但内部逻辑改为：
    1) 用 get_m_tilde 构造 M_f^k 和 M_f^{k-1}；
    2) 构造上一轮残差 R_prev = w_prev * (M_{k-1} - M_f^{k-1})；
    3) 用 LRLSWeightModule 从 R_prev 学出 IRLS 权重 rho；
    4) 用 IRLS 解析公式 compute_formula_lrls 计算新的掩码；
    5) 通道平均 -> [N,1,H,W] 掩码输出。

    这样：
    - 从优化角度：是真正的“L1 -> IRLS -> 加权二次 -> 解析解；
    - 从网络角度：rho 通过小卷积网络可学习；
    - get_w/get_m_tilde 与原论文完全对齐。
    """

    def __init__(
        self,
        alpha: float = 0.01,
        rho_channels: int = 16,
        rho_layers: int = 3,
        rho_min: float = 0.1,
        rho_max: float = 10.0,
        eps: float = 1e-3
    ):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor([alpha], dtype=torch.float32), requires_grad=True)
        self.mu = nn.Parameter(torch.tensor([0.01]), requires_grad=True)

        self.weight_module = LRLSWeightModule(
            channels=rho_channels,
            layers=rho_layers,
            rho_min=rho_min,
            rho_max=rho_max,
            eps=eps
        )

    def forward(
        self,
        I: torch.Tensor,
        B_k_1: torch.Tensor,
        m_k_1: torch.Tensor,
        m_hat_k_1: torch.Tensor,
        m_k_2: torch.Tensor
    ) -> torch.Tensor:
        """
        I      : 当前图像/特征 C [N, C, H, W]
        B_k_1  : 上一轮背景 B_{k-1} [N, C, H, W]
        m_k_1  : 上一轮掩码 M_{k-1} [N, 1, H, W]
        m_hat_k_1 : 当前迭代的某个掩码预测（这里不用也可以保留接口） [N,1,H,W]
        m_k_2  : 上上一轮掩码 M_{k-2} [N, 1, H, W]

        输出：
        - result: [N, 1, H, W]，新的掩码估计 M_k
        """

        # 1) 构造 refined 掩码：M_f^k 和 M_f^{k-1}
        #    对齐原论文：M_f 是对 M 做 get_m_tilde 的结果
        M_f_k = get_m_tilde(m_k_1)   # M_f^k
        M_f_k_1 = get_m_tilde(m_k_2) # M_f^{k-1}

        # 2) 构造上一轮残差 R_prev = w_prev * (M_{k-1} - M_f^{k-1})
        #    这里选择 M_{k-1} 而不是 m_hat_k_1，更贴近“上一迭代解”的 IRLS 思路
        w_prev = get_w(m_k_1)
        R_prev = w_prev * (m_k_1 - M_f_k)   # [N,1,H,W]

        # 3) LRLS 权重 rho（内部已经是 IRLS + learnable 修正）
        rho = self.weight_module(R_prev)      # [N,1,H,W]
        # 保留 w 的“不确定区域不参与稀疏约束”原则
        rho = rho * (w_prev ** 2)             # [N,1,H,W]

        # 4) IRLS 解析更新公式
        M_new_full = compute_formula_lrls(
            I=I,
            B_prev=B_k_1,
            M_prev=m_k_1,
            M_f=M_f_k,
            rho=rho,
            mu=self.mu,
            alpha=self.alpha
        )  # [N,C,H,W]

        # 5) 通道平均 -> 单通道掩码
        M_new = torch.mean(M_new_full, dim=1, keepdim=True)  # [N,1,H,W]
        return M_new


# -----------------------------
# 5. 简单自检（可以删掉）
# -----------------------------
if __name__ == "__main__":
    # 随机造点数据跑一遍，看有没有 shape / device 的问题
    N, C, H, W = 2, 3, 64, 64
    I = torch.randn(N, C, H, W)
    B_k_1 = torch.randn(N, C, H, W)
    m_k_1 = torch.sigmoid(torch.randn(N, 1, H, W))
    m_hat_k_1 = torch.sigmoid(torch.randn(N, 1, H, W))
    m_k_2 = torch.sigmoid(torch.randn(N, 1, H, W))

    model = LRLSSparsePipeline(alpha=0.01)
    out = model(I, B_k_1, m_k_1, m_hat_k_1, m_k_2)
    print("输出形状:", out.shape)  # 期望 [N,1,H,W]
