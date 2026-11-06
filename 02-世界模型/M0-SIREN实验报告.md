# M0 SIREN Phase alpha 实验报告

**日期**: 2026-05-06
**硬件**: CPU (PyTorch 2.11.0+cpu)
**模型**: SIREN 5层×256, ω₀=30, 265,220 参数

## 摘要

M0 SIREN 神经场原型通过两个基准实验，验证了 SIREN 架构在简单 3D 场景中的有效性。核心发现：视角多样性是神经场训练的关键。

## 实验结果

### E2.1a: 1-Ball (目标 PSNR > 35dB)

| 指标 | 值 |
|------|------|
| 最佳 PSNR | 36.56 dB |
| 状态 | ✅ PASS |
| 训练时间 | 462.3s (7.7min) |
| 参数量 | 265,220 |

**训练曲线**:
- Epoch 1: PSNR=6.9dB (3.3s/3epochs)
- Epoch 3: PSNR=11.8dB (3.5s/3epochs)
- Epoch 6: PSNR=11.9dB (3.1s/3epochs)
- Epoch 9: PSNR=17.7dB (3.0s/3epochs)
- Epoch 12: PSNR=25.0dB (3.4s/3epochs)
- Epoch 15: PSNR=28.3dB (3.4s/3epochs)
- Epoch 18: PSNR=30.3dB (3.5s/3epochs)
- Epoch 21: PSNR=30.6dB (3.5s/3epochs)
- Epoch 24: PSNR=31.0dB (3.5s/3epochs)
- Epoch 27: PSNR=32.9dB (3.5s/3epochs)
- Epoch 30: PSNR=34.0dB (3.6s/3epochs)
- Epoch 33: PSNR=36.6dB (3.5s/3epochs)

### E2.1b: 3-Ball (目标 PSNR > 30dB)

| 指标 | 值 |
|------|------|
| 最佳 PSNR | 30.63 dB |
| 状态 | ✅ PASS |
| 训练时间 | 531.6s (8.9min) |
| 参数量 | 265,220 |

**训练曲线**:
- Epoch 1: PSNR=7.0dB (3.6s/3epochs)
- Epoch 3: PSNR=9.0dB (3.4s/3epochs)
- Epoch 6: PSNR=11.2dB (3.3s/3epochs)
- Epoch 9: PSNR=14.5dB (3.1s/3epochs)
- Epoch 12: PSNR=18.2dB (3.0s/3epochs)
- Epoch 15: PSNR=22.1dB (3.2s/3epochs)
- Epoch 18: PSNR=24.5dB (3.9s/3epochs)
- Epoch 21: PSNR=26.3dB (3.7s/3epochs)
- Epoch 24: PSNR=26.5dB (3.7s/3epochs)
- Epoch 27: PSNR=27.1dB (3.7s/3epochs)
- Epoch 30: PSNR=28.2dB (3.5s/3epochs)
- Epoch 33: PSNR=28.8dB (3.5s/3epochs)
- Epoch 36: PSNR=29.9dB (3.0s/3epochs)
- Epoch 39: PSNR=30.6dB (3.5s/3epochs)

## 迭代历程

### v1 失败 (PSNR=5.6dB)
- **根因**: 训练光线从随机空间点指向场景中心，与评估时的相机光线分布完全不匹配
- **现象**: Loss=0.0005 极低但 PSNR=5.6dB，模型找到退化解（预测全黑）

### v2 突破 (Epoch 5 PSNR=33.3dB)
- **改进**: 训练光线从相机采样 + 场景查询监督
- **问题**: 单一训练视角导致过拟合，PSNR 在峰值后下降到 8.4dB

### v3 成功
- **核心改进**: 每个 epoch 使用多个不同相机视角 (n_views=3)
- **优化策略**: 学习率 5e-4 + ReduceLROnPlateau + 早停 (patience=15)

## 技术洞察

1. **视角多样性是神经场训练的关键** — 单一视角会导致模型记住该视角而非学习 3D 结构
2. **早停防止过拟合** — PSNR 在达到峰值后可能下降
3. **SIREN 架构验证** — 5 层、256 hidden、ω₀=30 的配置在 CPU 上可有效训练
4. **推理时间**: ~3.5s/64x64 图像 (CPU)，需 GPU 加速才能达到 <10ms 目标

## 后续实验

- E2.1c: 3-Ball + Slot Attention (目标 ARI>0.9)
- E2.1d: µP 可扩展性测试
- E2.2: 因果 Pong (S 级)

---

*M0 SIREN Phase alpha 实验报告 — 产出者: 锻智 NeuroForge*
