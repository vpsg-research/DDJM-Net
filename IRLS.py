import torch
import torch.nn as nn
import torch.nn.functional as F




def get_w(prob_map: torch.Tensor) -> torch.Tensor:

    assert prob_map.dim() == 4, "prob_map 应为 [N, C, H, W]"
    zero = prob_map.new_zeros(1)
    one = prob_map.new_ones(1)
    mask = (prob_map > 0.4) & (prob_map < 0.6)
    w = torch.where(mask, zero, one)
    return w


def get_m_tilde(prob_map: torch.Tensor) -> torch.Tensor:

    assert prob_map.dim() == 4, "prob_map 应为 [N, C, H, W]"
    low_val = prob_map.new_full((1,), 0.1)
    high_val = prob_map.new_full((1,), 0.9)

    out = torch.where((prob_map > 0.1) & (prob_map < 0.4),
                      low_val, prob_map)
    out = torch.where((out > 0.6) & (out < 0.9),
                      high_val, out)
    return out




class LRLSWeightModule(nn.Module):


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


def compute_formula_lrls(
    I: torch.Tensor,
    B_prev: torch.Tensor,
    M_prev: torch.Tensor,
    M_f: torch.Tensor,
    rho: torch.Tensor,
    mu: torch.Tensor,
    alpha: float
) -> torch.Tensor:
    

    assert I.dim() == 4 and B_prev.dim() == 4
    assert M_prev.dim() == 4 and M_prev.size(1) == 1
    assert M_f.dim() == 4 and M_f.size(1) == 1
    assert rho.dim() == 4 and rho.size(1) == 1

    I_sq = I ** 2  # [N, C, H, W]


    if rho.size(1) == 1 and I_sq.size(1) > 1:
        rho_expanded = rho.expand(-1, I_sq.size(1), -1, -1)
    else:
        rho_expanded = rho

    denominator = I_sq + mu + alpha * rho_expanded
    numerator = (
        I_sq - I * B_prev
        + mu * M_prev
        + alpha * rho_expanded * M_f
    )

    M_new_full = numerator / (denominator + 1e-6)
    return M_new_full




class LRLSSparsePipeline(nn.Module):
    

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
    


        M_f_k = get_m_tilde(m_k_1)   # M_f^k
        M_f_k_1 = get_m_tilde(m_k_2) # M_f^{k-1}


        w_prev = get_w(m_k_1)
        R_prev = w_prev * (m_k_1 - M_f_k)   # [N,1,H,W]


        rho = self.weight_module(R_prev)      # [N,1,H,W]
 
        rho = rho * (w_prev ** 2)             # [N,1,H,W]

        M_new_full = compute_formula_lrls(
            I=I,
            B_prev=B_k_1,
            M_prev=m_k_1,
            M_f=M_f_k,
            rho=rho,
            mu=self.mu,
            alpha=self.alpha
        )  # [N,C,H,W]

        
        M_new = torch.mean(M_new_full, dim=1, keepdim=True)  # [N,1,H,W]
        return M_new


# -----------------------------
# 5. 简单自检（可以删掉）
# -----------------------------
if __name__ == "__main__":
 
    N, C, H, W = 2, 3, 64, 64
    I = torch.randn(N, C, H, W)
    B_k_1 = torch.randn(N, C, H, W)
    m_k_1 = torch.sigmoid(torch.randn(N, 1, H, W))
    m_hat_k_1 = torch.sigmoid(torch.randn(N, 1, H, W))
    m_k_2 = torch.sigmoid(torch.randn(N, 1, H, W))

    model = LRLSSparsePipeline(alpha=0.01)
    out = model(I, B_k_1, m_k_1, m_hat_k_1, m_k_2)
    print("输出形状:", out.shape)  # 期望 [N,1,H,W]
