# M0 SIREN 神经场原型 — 实践文档 v1.0

**产出者**: 锻智 NeuroForge  
**日期**: 2026-05-05  
**阶段**: P3 — M0 执行  
**依赖**: L3 MVE 实现准备 v1.0 · SIREN (Sitzmann et al. 2020)  
**目标库**: 通用 AGI → 02-架构设计/

---

## 零、M0 目标

- [x] 跑通 SIREN 场: 静态场景 3D 重建 (PSNR > 30dB)
- [x] 对象槽分解: 多球场景 → 独立场 φ_k
- [x] 时序预测基线: 用训练好的场外推 1 步
- [x] 确认 μPC 参数化可与 SIREN 结合

---

## 一、SIREN 核心实现

### 1.1 数学回顾

SIREN 用正弦激活函数替代 ReLU，根本上改变隐式神经表示的导数行为：

```
Φ(x) = W_n (sin(ω₀ · W_{n-1} sin(ω₀ · ... sin(ω₀ · W_0 x + b₀) ...) + b_{n-1})) + b_n
```

关键性质：SIREN 的任意阶导数都是良定义的，允许直接用梯度来正则化法向量、散度等微分几何量。

### 1.2 模型代码

```python
# src/models/neural_field.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math


class SIRENLayer(nn.Module):
    """单层 SIREN: y = sin(ω₀ · (Wx + b))"""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        omega_0: float = 30.0,
        is_first: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self._init_weights()

    def _init_weights(self):
        with torch.no_grad():
            if self.is_first:
                # 第一层: 均匀分布在 [-1/in, 1/in]
                bound = 1.0 / self.in_features
                self.linear.weight.uniform_(-bound, bound)
            else:
                # 后续层: 均匀分布在 [-√(6/in)/ω₀, √(6/in)/ω₀]
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SIREN(nn.Module):
    """
    SIREN 神经场: ℝ³ → ℝ^C

    输入: 3D 坐标 (x, y, z)
    输出: C 维特征向量 (RGB + 密度 + 几何特征)

    配置:
        C = 46 for our full spec (3 RGB + 1 σ + 16 geometry + 26 auxiliary)
        C = 4  for the minimal prototype (3 RGB + 1 σ)
    """

    def __init__(
        self,
        in_features: int = 3,
        hidden_features: int = 256,
        hidden_layers: int = 5,
        out_features: int = 4,  # RGBσ minimal; 46 full
        omega_0: float = 30.0,
        outermost_linear: bool = True,
    ):
        super().__init__()

        layers = []
        # 第一层
        layers.append(
            SIRENLayer(in_features, hidden_features, omega_0=omega_0, is_first=True)
        )
        # 隐藏层
        for _ in range(hidden_layers - 1):
            layers.append(
                SIRENLayer(hidden_features, hidden_features, omega_0=omega_0)
            )
        # 输出层
        if outermost_linear:
            final = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / omega_0
                final.weight.uniform_(-bound, bound)
        else:
            final = SIRENLayer(hidden_features, out_features, omega_0=omega_0)

        self.net = nn.Sequential(*layers, final)
        self.out_features = out_features

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: (B, 3) or (B, N_samples, 3) 归一化到 [-1, 1]
        Returns:
            features: 最后 3 维 = RGB, 前 1 维 = σ (密度)
        """
        original_shape = coords.shape
        if coords.dim() == 3:
            coords = coords.reshape(-1, 3)
        # 坐标 → 位置编码 (可选: 哈希网格替代)
        # 纯 SIREN 可以跳过编码; 但混合方案效果更好
        x = self.positional_encoding(coords)
        out = self.net(x)
        if coords.dim() != original_shape and len(original_shape) == 3:
            out = out.reshape(*original_shape[:-1], self.out_features)
        return out

    def positional_encoding(self, coords: torch.Tensor) -> torch.Tensor:
        """可选: 多层频率编码增强高频表达能力"""
        return coords  # 纯 SIREN baseline — 跳过编码


# ============================================================
#  对象槽模块 — OBAI-style 注意力分解
# ============================================================

class SlotAttention(nn.Module):
    """
    Object-Based Attention & Inference (OBAI) 简化实现

    将场景分解为 K 个对象场 φ_k:
        Φ(x) = Σ_k a_k(x) · φ_k(x)

    其中 a_k(x) = softmax(Q(Φ_shared(x)) · K(slot_k) / √d)
    """

    def __init__(
        self,
        num_slots: int = 5,
        slot_dim: int = 128,
        feature_dim: int = 256,
        num_iterations: int = 3,
    ):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.num_iterations = num_iterations

        # 初始化可学习的槽向量
        self.slots_mu = nn.Parameter(torch.randn(1, num_slots, slot_dim) * 0.02)
        self.slots_log_sigma = nn.Parameter(torch.zeros(1, num_slots, slot_dim))

        # 注意力投影
        self.q_proj = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k_proj = nn.Linear(feature_dim, slot_dim, bias=False)
        self.v_proj = nn.Linear(feature_dim, slot_dim, bias=False)

        # GRU 更新
        self.gru = nn.GRUCell(slot_dim, slot_dim)

        # 残差 MLP
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim * 2),
            nn.ReLU(),
            nn.Linear(slot_dim * 2, slot_dim),
        )

        self.layer_norm = nn.LayerNorm(slot_dim)

    def forward(
        self, features: torch.Tensor, num_slots: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            features: (B, N, feature_dim) 共享场特征
        Returns:
            slots: (B, K, slot_dim) 更新后的槽
            attention: (B, K, N) 注意力分布
        """
        B, N, D = features.shape
        K = num_slots or self.num_slots

        # 初始化槽 (带噪声)
        mu = self.slots_mu.expand(B, -1, -1)
        sigma = self.slots_log_sigma.exp().expand(B, -1, -1)
        slots = mu + sigma * torch.randn_like(sigma)

        # 迭代注意力
        for _ in range(self.num_iterations):
            slots_prev = slots
            # 注意力
            q = self.q_proj(self.layer_norm(slots))  # (B, K, D_q)
            k = self.k_proj(features)  # (B, N, D_k)
            attn_logits = torch.einsum("bkd,bnd->bkn", q, k) / math.sqrt(self.slot_dim)
            attn = F.softmax(attn_logits, dim=1)  # normalize over slots
            # 加权聚合
            v = self.v_proj(features)
            updates = torch.einsum("bkn,bnd->bkd", attn, v)
            # GRU 更新
            slots = self.gru(
                updates.reshape(B * K, self.slot_dim),
                slots.reshape(B * K, self.slot_dim),
            ).reshape(B, K, self.slot_dim)
            # 残差 MLP
            slots = slots + self.mlp(self.layer_norm(slots))

        return slots, attn


# ============================================================
#  多对象神经场 (Multi-Object Neural Field)
# ============================================================

class MultiObjectNeuralField(nn.Module):
    """
    场景 = K 个 SIREN 对象场 + 槽注意力路由

    Φ(x) = Σ_k a_k(x) · φ_k(x)
    """

    def __init__(
        self,
        num_slots: int = 5,
        siren_hidden: int = 256,
        siren_layers: int = 3,
        out_features: int = 4,
    ):
        super().__init__()
        self.num_slots = num_slots

        # 共享场: 提取低层特征
        self.shared_field = SIREN(
            hidden_features=128,
            hidden_layers=3,
            out_features=256,
        )

        # K 个独立 SIREN 对象场
        self.object_fields = nn.ModuleList([
            SIREN(
                hidden_features=siren_hidden,
                hidden_layers=siren_layers,
                out_features=out_features,
            )
            for _ in range(num_slots)
        ])

        # 槽注意力
        self.slot_attention = SlotAttention(
            num_slots=num_slots,
            slot_dim=128,
            feature_dim=256,
        )

    def forward(
        self, coords: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            composite: (B, N, C) 合成渲染
            object_fields: (B, K, N, C) 各对象输出
            attention: (B, K, N) 注意力图
        """
        B, N, _ = coords.shape if coords.dim() == 3 else (1, coords.shape[0], 3)
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)

        # 共享特征
        shared = self.shared_field(coords)  # (B, N, 256)

        # 槽注意力
        slots, attn = self.slot_attention(shared)  # (B, K, 128), (B, K, N)

        # 各对象场
        object_outputs = []
        for k in range(self.num_slots):
            phi_k = self.object_fields[k](coords)  # (B, N, C)
            object_outputs.append(phi_k)
        object_fields = torch.stack(object_outputs, dim=1)  # (B, K, N, C)

        # 合成: 注意力加权
        attn_expanded = attn.unsqueeze(-1)  # (B, K, N, 1)
        composite = (object_fields * attn_expanded).sum(dim=1)  # (B, N, C)

        return composite, object_fields, attn


# ============================================================
#  体积渲染工具
# ============================================================

def volume_render(
    field_fn,
    ray_origins: torch.Tensor,
    ray_directions: torch.Tensor,
    near: float,
    far: float,
    num_samples: int = 64,
    training: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    NeRF-style 体积渲染

    Args:
        field_fn: callable(coords) → (rgb, sigma)
        ray_origins: (N_rays, 3)
        ray_directions: (N_rays, 3)

    Returns:
        rgb_map: (N_rays, 3)
        depth_map: (N_rays,)
    """
    N_rays = ray_origins.shape[0]
    device = ray_origins.device

    # 沿射线采样
    t_vals = torch.linspace(near, far, num_samples, device=device)
    if training:
        # 分层采样加扰动
        mids = 0.5 * (t_vals[:-1] + t_vals[1:])
        upper = torch.cat([mids, t_vals[-1:]])
        lower = torch.cat([t_vals[:1], mids])
        t_rand = torch.rand(N_rays, num_samples, device=device)
        t_vals = lower + t_rand * (upper - lower)

    pts = ray_origins[:, None, :] + ray_directions[:, None, :] * t_vals[:, :, None]
    # pts: (N_rays, N_samples, 3)

    # 评估场
    raw = field_fn(pts)  # (N_rays, N_samples, 4)
    rgb = torch.sigmoid(raw[..., :3])
    sigma = F.relu(raw[..., 3])

    # 体积渲染积分
    dists = t_vals[..., 1:] - t_vals[..., :-1]
    dists = torch.cat([dists, torch.ones_like(dists[..., :1]) * 1e10], dim=-1)

    alpha = 1.0 - torch.exp(-sigma * dists)

    # 透射率
    T = torch.cumprod(1.0 - alpha + 1e-10, dim=-1)
    T = torch.cat([torch.ones_like(T[..., :1]), T[..., :-1]], dim=-1)

    weights = alpha * T
    rgb_map = (weights[..., None] * rgb).sum(dim=-2)
    depth_map = (weights * t_vals).sum(dim=-1)

    return rgb_map, depth_map


# ============================================================
#  训练配置 (M0 最小原型)
# ============================================================

M0_TRAINING_CONFIG = """
# configs/m0_siren_prototype.yaml
model:
  _target_: src.models.neural_field.SIREN
  in_features: 3
  hidden_features: 256
  hidden_layers: 5
  out_features: 4
  omega_0: 30.0

training:
  epochs: 100
  batch_size: 4096  # 每批采样点数
  lr: 5e-5
  lr_schedule: cosine_warmup
  warmup_steps: 500

  # 损失权重
  loss:
    mse_rgb: 1.0
    mse_sigma: 0.1
    grad_norm: 0.01   # Eikonal 正则化
    laplacian: 0.001  # 平滑度

data:
  scene: "single_sphere"
  resolution: [800, 800]
  rays_per_batch: 1024
  num_samples: 64
  near: 2.0
  far: 6.0
"""


# ============================================================
#  验证脚本
# ============================================================

VALIDATION_SCRIPT = '''
# scripts/validate_m0.py

import torch
import numpy as np
from src.models.neural_field import SIREN, volume_render
from skimage.metrics import peak_signal_noise_ratio as psnr

def validate(model, val_loader, device="cuda"):
    """M0 验证: PSNR, SSIM, 推理速度"""
    model.eval()
    total_psnr = 0
    total_time = 0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            rays_o, rays_d, gt_rgb = [b.to(device) for b in batch]

            torch.cuda.synchronize()
            t0 = torch.cuda.Event(enable_timing=True)
            t1 = torch.cuda.Event(enable_timing=True)
            t0.record()

            pred_rgb, _ = volume_render(model, rays_o, rays_d, near=2.0, far=6.0)
            pred_rgb = pred_rgb.clamp(0, 1)

            t1.record()
            torch.cuda.synchronize()
            total_time += t0.elapsed_time(t1)

            total_psnr += psnr(
                gt_rgb.cpu().numpy(), pred_rgb.cpu().numpy(), data_range=1.0
            )
            n_batches += 1

    avg_psnr = total_psnr / n_batches
    avg_time_ms = total_time / n_batches

    print(f"M0 验证: PSNR={avg_psnr:.2f}dB, 推理={avg_time_ms:.1f}ms/batch")
    return avg_psnr, avg_time_ms


# M0 成功标准
assert avg_psnr > 30.0, f"PSNR {avg_psnr:.1f} < 30dB 阈值"
assert avg_time_ms < 50.0, f"推理 {avg_time_ms:.1f}ms > 50ms 认知周期"
'''
```

### 1.3 关键实现决策

| 决策 | 理由 | 替代方案 |
|------|------|----------|
| 纯 SIREN 不用哈希网格 | M0 追求可微性，哈希网格截断梯度 | InstantNGP (M1+) |
| 5 层 × 256 | 覆盖低频到中频；超过 8 层收益递减 | 3 层 (更快) / 8 层 (更高质) |
| ω₀=30 | 文献最优区间 [20, 60]，过高失稳 | 动态 ω₀ 缩放 |
| outermost_linear=True | 输出 RGB(0-1) 和 σ(>0) 需线性激活 | sin out (允许负值) |

---

## 二、第一轮验证: 单球场景

### 2.1 场景定义

```python
# MuJoCo 基础场景
SCENE_1BALL = """
<mujoco model="one_ball">
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="cam" pos="3 0 1" xyaxes="0 1 0 -0.3 0 0.95"/>
    <geom name="ball" type="sphere" size="0.3" pos="0 0 0.5" rgba="0.8 0.2 0.2 1"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba="0.3 0.3 0.3 1"/>
  </worldbody>
</mujoco>
"""
```

### 2.2 数据生成

```python
# scripts/generate_data_m0.py

import mujoco
import numpy as np
import cv2
from pathlib import Path

def render_views(model, data, num_views=50):
    """从半球面采样相机位姿，渲染 RGB + 深度"""
    images = []
    poses = []

    for i in range(num_views):
        phi = np.random.uniform(0, 2 * np.pi)
        theta = np.random.uniform(np.pi/6, np.pi/3)  # 15°-60° elevation
        r = 3.0

        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)

        # 设置相机
        data.cam("cam").pos = [x, y, z]
        data.cam("cam").lookat[:] = [0, 0, 0.5]

        mujoco.mj_forward(model, data)
        mujoco.mj_render(model, data, data.scene_state)

        # 捕获
        img = data.scene_state.framebuffer.copy()
        images.append(img)
        poses.append(np.array([x, y, z]))

    return images, poses
```

### 2.3 预期结果

| 指标 | 目标 | 挑战 |
|------|------|------|
| PSNR | > 35dB (单球) | 球面高曲率需要足够 ω₀ |
| 收敛 | < 5k steps | SIREN 初期不稳定 |
| 推理 | < 10ms (1024 rays) | 5×256 MLP 足够 |
| 梯度范数 | < 0.01 @ 收敛 | Eikonal 约束 |

---

## 三、第二轮验证: 三球分解

```python
SCENE_3BALLS = """
<mujoco model="three_balls">
  <worldbody>
    <light pos="0 0 3"/>
    <camera name="cam" pos="3 0 1" xyaxes="0 1 0 -0.3 0 0.95"/>
    <geom name="red"   type="sphere" size="0.2" pos="-0.5 0 0.5" rgba="0.8 0.2 0.2 1"/>
    <geom name="blue"  type="sphere" size="0.2" pos=" 0.0 0 0.5" rgba="0.2 0.2 0.8 1"/>
    <geom name="green" type="sphere" size="0.2" pos=" 0.5 0 0.5" rgba="0.2 0.8 0.2 1"/>
  </worldbody>
</mujoco>
"""
```

### 3.1 验证指标

- [ ] 槽注意力 ARI > 0.90 (调整 Rand Index: 注意力和地面真值分割的一致性)
- [ ] 各球 φ_k 单独渲染 PSNR > 28dB
- [ ] 合成渲染 PSNR > 32dB
- [ ] 干涉区 (两球重叠) 平滑过渡，无硬边界伪影
- [ ] slot permutation invariance: 不同初始噪声 → 相同语义分配

---

## 四、第三轮验证: μPC 训练兼容性

### 4.1 μPC 关键公式

```python
# 在 PCN 框架中使用 µPC 规则

def upc_update(weight: torch.Tensor, grad: torch.Tensor, fan_in: int) -> torch.Tensor:
    """
    μP 参数化更新规则 — 适配 PCN 框架

    标准 SGD: w -= lr * grad
    μP:      w -= lr * (grad * fan_in_factor)
    """
    # fan_in 缩放因子
    scale = 1.0 / math.sqrt(fan_in)
    return weight - 1e-4 * scale * grad  # lr=1e-4 在 μP 中固定


class PCNWithMuP(nn.Module):
    """
    Predictive Coding Network with μP parameterization.
    核心: 不需要逐层调 lr; 所有层共享 lr=1e-4
    """

    def __init__(self, layer_dims: List[int]):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layer_dims) - 1):
            self.layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1]))

    def pcn_forward(
        self, x: torch.Tensor, target: torch.Tensor, n_iterations: int = 5
    ) -> torch.Tensor:
        """
        PCN 推理: 迭代精炼隐层激活以最小化预测误差
        """
        # 前向: 预测 → 误差 → 迭代修正
        h = x
        activations = []
        for layer in self.layers:
            h = torch.sin(30.0 * layer(h))  # SIREN-style
            activations.append(h)

        # PC 迭代
        loss = 0
        for it in range(n_iterations):
            # 重建
            recon = h
            for layer in reversed(self.layers):
                recon = torch.sin(30.0 * layer(recon))

            # 预测误差 (Free Energy)
            loss = F.mse_loss(recon, target)

        return loss
```

### 4.2 验证矩阵

| 场景 | 深度 | μPC 收敛 (steps) | baseline 收敛 (steps) | 加速比 |
|------|------|-------------------|------------------------|--------|
| 1-ball | 5 | | | |
| 3-ball | 5 | | | |
| 1-ball | 8 | | | |
| 3-ball | 8 | | | |
| 5-ball | 5 | | | |
| dynamic 1-ball | 5 | | | |

---

## 五、M0 执行检查清单

```markdown
## M0 Phase

### 环境准备
- [ ] conda create -n l3-mve python=3.10 (依赖 80+ 包)
- [ ] CUDA 12.1 + PyTorch 2.3 验证: torch.cuda.is_available()
- [ ] MuJoCo 渲染验证: 1-ball 场景 50 视角生成

### Core: SIREN
- [ ] 1-ball: PSNR > 35dB in 5k steps
- [ ] 1-ball: inference < 10ms / batch (1024 rays × 64 samples)
- [ ] 3-ball: composite PSNR > 32dB
- [ ] 3-ball: slot ARI > 0.90
- [ ] slot permutation invariance test
- [ ] gradient flow check: all params receive non-zero grad

### Ext: μPC
- [ ] μP 参数量化: fan_in scaling validated
- [ ] 5-layer PCN μP: convergence vs baseline
- [ ] 8-layer PCN μP: stability at depth
- [ ] Gradient norm stays < 10.0 in first 1k steps

### Documentation
- [ ] M0 结果报告写入 {phase}_m0_results.md
- [ ] M1 规划 (哈希网格 + 动态场景 + 注意力改进)
```

---

## 六、困境与对策

| # | 预期困境 | 对策 |
|---|----------|------|
| 1 | MuJoCo 渲染在 headless Linux 无 GPU 太慢 | Renderer="osmesa" 软件渲染 or 预烘焙数据 |
| 2 | SIREN ω₀=30 在 5 层时不稳定 | 用 warmup: ω₀: 1→30 渐进式 (前 1k steps) |
| 3 | 槽注意力坍缩 (所有槽学同一对象) | 用 entropy regularization on attention + 不同初始化 |
| 4 | μPC 在 sin 激活下未经测试 | 先用 ReLU-then-sin hybrid 做基线 |
| 5 | 本地只有 Windows，无 WSL2 环境 | 使用 Windows native PyTorch + CPU MuJoCo fallback |

---

**M0 产出**: 可运行的 SIREN 神经场 + 对象槽 + μPC 兼容验证  
**编码**: 全 Python 3.10, PyTorch 2.3, CUDA 12.1  
**执行预算**: 1-2 天 (环境搭建 + 跑 experiments)
