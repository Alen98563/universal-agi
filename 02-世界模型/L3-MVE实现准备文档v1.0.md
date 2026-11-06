# L3 神经场世界模型 — MVE 实现准备文档 v1.0

**产出者**: 锻智 NeuroForge  
**日期**: 2026-05-04  
**阶段**: P2 — L3 最小可行实验工程准备  
**依赖**: 消息协议 v1.0 + L3 原型设计 + 白皮书 v0.2  
**目标库**: 通用 AGI → 02-架构设计/

---

## 一、环境与依赖

### 1.1 硬件要求

| 组件 | 最低 | 推荐 |
|------|------|------|
| GPU | RTX 3060 12GB | RTX 4090 24GB / A100 40GB |
| RAM | 32GB | 64GB |
| 存储 | 100GB SSD | 500GB NVMe |
| OS | Ubuntu 22.04 / Windows 11 WSL2 | Ubuntu 22.04 |

### 1.2 Python 环境

```bash
# Python 3.10+
conda create -n l3-mve python=3.10
conda activate l3-mve

# 核心依赖
pip install torch==2.3.0 torchvision==0.18.0 --index-url https://download.pytorch.org/whl/cu121
pip install lightning==2.2.0
pip install wandb==0.17.0

# 神经场
pip install nerfstudio==1.1.0
pip install tinycudann==1.7  # 加速哈希编码

# 物理仿真 (数据生成)
pip install mujoco==3.1.0
pip install gymnasium==0.29.0

# 因果
pip install causallearn==0.1.3
pip install dowhy==0.11.0
pip install lingam==1.8.0

# 工具
pip install einops==0.8.0
pip install hydra-core==1.3.0
pip install rich==13.0.0
pip install pytest==8.0.0
pip install black==24.0.0
pip install ruff==0.4.0
```

### 1.3 仓库结构

```
l3-world-model/
├── README.md
├── pyproject.toml
├── configs/
│   ├── base.yaml                 # 基础超参数
│   ├── phase1_pretrain.yaml      # Phase 1 感知预训练
│   ├── phase2_dynamics.yaml      # Phase 2 动力学微调
│   ├── phase3_metatrain.yaml     # Phase 3 元学习泛化
│   └── mve_causal_pong.yaml     # MVE 因果 Pong
├── src/
│   ├── __init__.py
│   ├── models/
│   │   ├── __init__.py
│   │   ├── neural_field.py       # SIREN / InstantNGP 神经场
│   │   ├── object_slot.py        # OBAI 对象槽 + 注意力路由
│   │   ├── dynamics.py           # 动力学预测器 (Transformer latent)
│   │   ├── predictive_coding.py  # PCN 训练循环 (µPC 参数化)
│   │   └── world_model.py        # WorldModel 总装
│   ├── data/
│   │   ├── __init__.py
│   │   ├── dataset.py            # 数据加载器
│   │   ├── simulation.py         # MuJoCo 场景生成
│   │   ├── causal_pong.py        # 因果 Pong 环境
│   │   └── augment.py            # 数据增强
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py            # Lightning Trainer 封装
│   │   ├── losses.py             # 损失函数 (F, A4)
│   │   └── optimizer.py          # µPC 参数化优化器
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py            # 评估指标 (重建/预测/因果)
│   │   ├── benchmark.py          # vs DreamerV3 基准测试
│   │   └── visualization.py      # 场可视化工具
│   ├── causal/
│   │   ├── __init__.py
│   │   ├── discovery.py          # 因果发现 (ILS-CSL)
│   │   ├── compression.py        # 因果紧缩 (Abs-LiNGAM 桥)
│   │   └── do_calculus.py        # do-演算实现
│   └── protocols/
│       ├── __init__.py
│       ├── message.py            # 消息格式 (v1.0 规范)
│       ├── scheduler.py          # 认知周期调度器
│       └── router.py             # 注意力路由
├── tests/
│   ├── test_neural_field.py
│   ├── test_predictive_coding.py
│   ├── test_dynamics.py
│   ├── test_causal_compression.py
│   └── test_mve.py
├── scripts/
│   ├── generate_data.py          # 数据生成脚本
│   ├── train_phase1.py
│   ├── train_phase2.py
│   ├── train_phase3.py
│   ├── eval_mve.py              # MVE 评估
│   └── benchmark_dreamerv3.py    # 对比实验
└── notebooks/
    ├── 01_field_viz.ipynb        # 神经场可视化
    ├── 02_causal_inspection.ipynb # 因果图检视
    └── 03_mve_results.ipynb     # MVE 结果汇总
```

---

## 二、数据管道

### 2.1 数据源规格

| 阶段 | 数据 | 规模 | 生成方式 |
|------|------|------|----------|
| Phase 1 感知 | 单场景多视角 | 10³ 场景 × 100 视角 | MuJoCo 渲染 + COLMAP 位姿估计 |
| Phase 2 动力学 | 物理交互视频 | 10⁴ 交互 × 64 帧 | MuJoCo 推/抓/放对象 |
| Phase 3 元学习 | 跨场景对象集 | 50 场景 × 20 对象类型 | 程序化对象生成 + 物理 |

### 2.2 因果 Pong 环境

```
causal_pong = {
  objects:     [ball_A, ball_B],
  geometry:    2D (可拉伸至 3D),
  dynamics:    弹性碰撞 + 摩擦力,
  actions:     [push(ball, direction, force)],
  causality:   ball_A 碰撞 → ball_B 轨迹改变,
  
  observation: 64×64×3 RGB 渲染,
  horizon:     100 steps × 16ms = 1.6s,
  
  train_episodes:  10,000,
  test_episodes:   1,000,
  
  因果变体:
    - Standard:  A→B 碰撞 (直接因果)
    - Hidden:    A 被遮挡 50% 时间 (需要世界模型维持对象)
    - Counterfactual: 初始条件微扰 → 不同结果 (do-演算评估)
}
```

### 2.3 数据格式

```python
@dataclass
class Frame:
    """单帧数据"""
    rgb: torch.Tensor          # [H, W, 3]  uint8
    depth: torch.Tensor        # [H, W]    float32
    segmentation: torch.Tensor # [H, W]    int32 (对象实例 ID)
    camera_pose: torch.Tensor  # [4, 4]    float32 (world→camera)
    timestamp: float           # 帧时间 (ms)
    
@dataclass
class Episode:
    """一个完整交互序列"""
    frames: List[Frame]        # 64-100 帧
    actions: List[Action]      # 施加的动作
    causal_graph: nx.DiGraph   # 因果地面真值 (评估用)
    metadata: Dict             # 场景描述
```

---

## 三、模型实现规格

### 3.1 神经场 — `NeuralField`

```python
class NeuralField(nn.Module):
    """
    φ(x, t): ℝ³ × ℝ → ℝ⁴⁶
    坐标基神经场 — SIREN 激活 + 位置编码
    """
    
    def __init__(self,
        d_in: int = 3,           # 空间坐标 (x,y,z)
        d_time: int = 1,         # 时间
        d_out: int = 46,         # 输出通道
        hidden: int = 256,
        n_layers: int = 4,
        omega_0: float = 30.0,   # SIREN 频率
        n_freqs: int = 10,       # 位置编码频率数
    ):
        ...
    
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        x: [B, N, 3]  空间坐标
        t: [B, N, 1]  时间
        
        返回: [B, N, 46]
        [0:4]   geometry (SDF + occupancy)
        [4:36]  semantics (CLIP embedding)
        [36:44] affordance
        [44:46] uncertainty
        """
        ...
    
    def predict_geometry(self, x: Tensor, t: Tensor) -> Tensor:
        """返回 [B, N, 4] — SDF + occupancy"""
        return self(x, t)[..., :4]
    
    def predict_semantics(self, x: Tensor, t: Tensor) -> Tensor:
        """返回 [B, N, 32] — 语义嵌入"""
        return self(x, t)[..., 4:36]
```

### 3.2 对象槽 — `ObjectSlot`

```python
class ObjectSlot(nn.Module):
    """
    每个对象槽独立建模一个对象场 φ_k
    注意力路由: 输入 → 最匹配的槽
    """
    
    def __init__(self, K: int = 16, d_query: int = 64):
        self.slots = nn.ModuleList([
            NeuralField() for _ in range(K)
        ])
        self.router = SlotAttentionRouter(K, d_query)
    
    def forward(self, x: Tensor, t: Tensor, G_t: Tensor):
        """
        x, t: 输入查询点
        G_t:  L5 全局态 (用于注意力查询)
        """
        # 1. 注意力路由
        alpha = self.router(G_t, x, self.slots)
        # alpha: [B, K] — 每个查询点分配给各槽的权重
        
        # 2. 各槽独立推理
        outputs = []
        for k in range(self.K):
            phi_k = self.slots[k](x, t)  # [B, N, 46]
            outputs.append(alpha[:, k:k+1, None] * phi_k)
        
        # 3. 融合: Φ(x,t) = Σ α_k·φ_k(x,t)
        return torch.stack(outputs).sum(dim=0)
```

### 3.3 动力学预测器 — `DynamicsPredictor`

```python
class DynamicsPredictor(nn.Module):
    """
    F: z_k^t → z_k^{t+Δt}
    Transformer over latent tokens
    """
    
    def __init__(self, d_latent: int = 128, n_layers: int = 2):
        self.tokenizer = nn.Linear(46, d_latent)     # φ→token
        self.transformer = nn.TransformerEncoder(...) # 2-layer
        self.decoder = nn.Linear(d_latent, 46)        # token→φ
        
    def forward(self, z: Tensor) -> Tensor:
        """
        z: [B, K, 46] — K 个对象的当前状态
        
        返回: [B, K, 46] — K 个对象的下一状态
        """
        tokens = self.tokenizer(z)          # [B, K, d_latent]
        tokens_next = self.transformer(tokens)  # self-attention
        return self.decoder(tokens_next)     # [B, K, 46]
```

### 3.4 预测编码训练循环 — `PCNTrainer`

```python
class PCNTrainer:
    """
    µPC 参数化的多层预测编码训练
    """
    
    def train_step(self, batch):
        # 1. 前馈 (Bottom-up ERROR↑)
        errors = []
        for layer in self.layers:
            prediction = self.predict(layer)
            actual = self.encode(layer, batch)
            error = actual - prediction
            errors.append(error)
        
        # 2. 反馈 (Top-down PRED↓)
        for layer in reversed(self.layers):
            prediction = self.imagine(layer)  # 从上层生成预测
            self.layers[layer].prior = prediction
        
        # 3. 自由能梯度 (A4)
        F = sum(w * precision * (e**2).sum() for w, precision, e in 
                zip(self.weights, self.precisions, errors))
        
        # 4. µPC 参数化更新
        # 学习率缩放: η/width → 保证极限宽度下的稳定训练
        F.backward()
        for p in self.parameters():
            p.grad *= self.lr / sqrt(self.width)
        
        return F.item()
```

---

## 四、训练配置

### 4.1 Phase 1 — 感知预训练

```yaml
# configs/phase1_pretrain.yaml
phase: 1
name: "l3-phase1-pretrain"

model:
  neural_field:
    type: "siren"
    d_out: 46
    hidden: 256
    n_layers: 4
    omega_0: 30.0
  slot_count: 16
  use_positional_encoding: true
  n_freqs_pos: 10
  n_freqs_time: 6

training:
  optimizer: "adamw"
  lr: 1e-4
  weight_decay: 0.01
  batch_size: 16
  accumulate_grad_batches: 4
  max_steps: 1_000_000
  
  losses:
    geometry: 
      type: "mse"
      weight: 1.0
    semantics:
      type: "cosine"
      weight: 0.5
    uncertainty:
      type: "calibration"
      weight: 0.1
      beta: 1.0  # meta-d' style

data:
  scenes: 1000
  views_per_scene: 100
  resolution: [256, 256]
  rays_per_batch: 1024
  
evaluation:
  every_n_steps: 5000
  metrics: ["psnr", "ssim", "lpips"]
  save_checkpoint: true
  
wandb:
  project: "l3-world-model"
  run: "phase1-pretrain"
```

### 4.2 Phase 2 — 动力学微调

```yaml
# configs/phase2_dynamics.yaml
phase: 2
name: "l3-phase2-dynamics"

model:
  load_pretrained: "phase1_best.ckpt"
  dynamics:
    d_latent: 128
    n_transformer_layers: 2
    n_heads: 8
    dropout: 0.1
  
  freeze_geometry: true  # 冻结几何头部
  train_semantics: true
  train_dynamics: true

training:
  lr: 5e-5                # 更低 — 微调
  batch_size: 8           # 场景数
  seq_len: 8              # 时间步数
  max_steps: 1_000_000
  
  losses:
    reconstruction:    # 光度 (standard NeRF)
      weight: 0.3
    dynamics_1step:    # 1步预测 MSE
      weight: 1.0
    dynamics_4step:    # 4步 rollout MSE
      weight: 0.5
    jepa_mask:         # VJEPA-style 遮掩预测
      weight: 0.3
      mask_ratio: 0.75
    causal_jepa:       # Causal-JEPA 遮掩
      weight: 0.2
      mask_object: true
      target_causal_direction: true

data:
  interactions: 10000
  frames_per_interaction: 64
  fps: 30

evaluation:
  horizon: [1, 4, 8, 16]
  metric: "mse_per_step"
```

### 4.3 MVE — 因果 Pong

```yaml
# configs/mve_causal_pong.yaml
name: "l3-mve-causal-pong"

model:
  load_pretrained: "phase2_best.ckpt"
  slot_count: 2           # 因果 Pong 仅两个对象
  freeze_encoder: false   # 从头训练 MVE

training:
  total_episodes: 10000
  warmup_episodes: 1000
  batch_size: 32
  lr: 3e-4
  max_steps: 200_000
  
  losses:
    action_conditional:   # P(Φ_{t+1} | Φ_t, action)
      weight: 1.0
    collision_prediction: # 碰撞是否发生 (BCE)
      weight: 0.5
    trajectory_mse:       # 位置预测
      weight: 0.3

data:
  environment: "causal_pong"
  variants: ["standard", "hidden", "counterfactual"]

baselines:
  - name: "dreamerv3"
    config: "dreamerv3_small"
  - name: "vjepa"
    config: "vjepa_video"

evaluation:
  steps: [1, 4, 16]
  metrics:
    - "position_mse"
    - "collision_accuracy"
    - "trajectory_mse"
    - "causal_direction_accuracy"
```

---

## 五、评估体系

### 5.1 评估指标

```python
def compute_metrics(pred, target, mode='full'):
    """
    三层评估矩阵
    """
    metrics = {}
    
    # A. 重建保真度
    metrics['psnr'] = 10 * log10(1.0 / mse(pred.rgb, target.rgb))
    metrics['ssim'] = ssim(pred.rgb, target.rgb)
    metrics['lpips'] = lpips(pred.rgb, target.rgb)
    metrics['depth_mae'] = mae(pred.depth, target.depth)
    
    # B. 物理预测精度
    for horizon in [1, 4, 8, 16]:
        # rollout 预测
        rollout = rollout_dynamics(model, target[0], horizon)
        metrics[f'pos_mse_{horizon}'] = mse(rollout.pos, target[:horizon].pos)
        metrics[f'collision_acc_{horizon}'] = accuracy(
            rollout.collision, target[:horizon].collision
        )
    
    # C. 因果涌现度量
    metrics['causal_direction'] = evaluate_causal_direction(
        model, causal_ground_truth
    )
    metrics['counterfactual_accuracy'] = evaluate_counterfactual(
        model, counterfactual_test_set
    )
    
    return metrics

def evaluate_causal_direction(model, ground_truth: nx.DiGraph):
    """
    Causal-JEPA 遮掩 → 涌现因果方向
    度量: 模型推断出的因果边与地面真值的 Jaccard 相似度
    """
    inferred = model.extract_causal_graph()
    jaccard = len(inferred.edges & ground_truth.edges) / \
              len(inferred.edges | ground_truth.edges)
    return jaccard

def evaluate_counterfactual(model, test_set):
    """
    反事实查询: "如果推对象A→ 对象B会怎样?"
    
    测试集: 100 组 (初始状态, 干预, 参考结果)
    模型预测 do(A=push) 的效果 → 与参考结果比较
    """
    accuracies = []
    for s0, intervention, reference in test_set:
        pred = model.predict_counterfactual(s0, intervention)
        accuracies.append(cosine_similarity(pred, reference))
    return np.mean(accuracies)
```

### 5.2 vs DreamerV3 基准测试

```python
def benchmark_vs_baselines():
    """
    因果 Pong — L3 PCN vs DreamerV3 vs V-JEPA
    
    预期结果 (假设):
    
    | 指标 | L3 PCN (ours) | DreamerV3 | V-JEPA |
    |------|--------------|-----------|--------|
    | 1-step MSE | 0.0012 | 0.0035 | 0.0028 |
    | 4-step MSE | 0.018 |  0.034  | 0.025 |
    | 16-step MSE | 0.12 | 0.31 | 0.22 |
    | Collision Acc | 0.94 | 0.76 | 0.82 |
    | Causal Direction | 0.87 | 0.45 | 0.62 |
    | Counterfactual | 0.71 | 0.38 | 0.51 |
    
    L3 PCN 在长程预测和因果度量上有显著优势。
    """
```

---

## 六、里程碑与进度追踪

| M# | 任务 | 输入 → 输出 | 预计 | 验收标准 |
|----|------|-------------|------|----------|
| **M0** | 神经场原型 | MuJoCo 单场景 → SIREN 场 | 2周 | PSNR > 30, SSIM > 0.9 vs NeRF baseline |
| **M1** | PCN 训练 | M0 场 → µPC 多层 | 4周 | 3+ PC 层稳定; F < baseline; 误差逐层递减 |
| **M2** | 动力学 F | M1 PCN + 交互数据 → 多步预测 | 4周 | 4-step MSE < DreamerV3; rollout 不爆炸 |
| **M3** | 对象槽 OBAI | M2 场 → 对象分解 | 3周 | 2+ 对象准确分割; 注意力路由正确 |
| **M4** | 因果涌现 | M3 槽 + Causal-JEPA → 因果图 | 4周 | Causal Direction > 0.8; Counterfactual > 0.6 |
| **M5** | MVE 集成 | M0-M4 → 因果 Pong | 2周 | 三项指标超过 DreamerV3 + V-JEPA |
| **M6** | L3 v1.0 | M5 报告 → 论文草稿 | — | 完整实验报告 + 可视化 |

### 当前状态: M0 准备就绪 — 等待执行

```
Phase 1 感知预训练 (M0):
  [ ] 安装依赖环境
  [ ] MuJoCo 场景生成 (generate_data.py)
  [ ] 训练 SIREN 场 (train_phase1.py)
  [ ] 评估 vs NeRF baseline
  [ ] Push to wandb + git tag M0
  
Phase 2 动力学微调 (M2):
  [ ] 物理交互数据生成
  [ ] µPC 参数化适配
  [ ] PCN 多层训练
  [ ] 动力学 Transformer 训练
  [ ] 评估 rollout vs DreamerV3
```

---

## 七、风险与缓解

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| µPC 在神经场上不稳定 | 中 | 高 | 回退到标准 PC + 降低层数 (min 2层) |
| 因果 Pong 太简单 → 无区分度 | 低 | 中 | 增加 Hidden + Counterfactual 变体 |
| COS 上传网络异常 | 中 | 低 | 重试 + 本地备份 |
| GPU 显存不足 (12GB) | 中 | 高 | 降低 batch + ray count + mixed precision |
| DreamerV3 难以复现 | 中 | 低 | 使用官方 checkpoint; MVE 独立评估即可 |

---

*L3 MVE 实现准备文档 v1.0 — P2 第二份产出。完整工程规格：环境、仓库结构、数据管道、模型实现、训练配置、评估体系、里程碑。*
