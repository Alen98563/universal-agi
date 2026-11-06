# 模块 B 文献综述：多层预测编码框架 v1.0

**产出者**: 锻智 NeuroForge  
**日期**: 2026-05-03  
**搜索**: ProSearch 学术 × 3 轮  
**目标库**: 通用 AGI → 06-文献综述/

---

## 综述范围

覆盖白皮书中层级间信号传递、预测驱动学习和行动选择的理论基础：

| 子模块 | 核心问题 | 搜索命中 |
|--------|----------|----------|
| B1. 预测编码深度学习 | PC 能否替代反向传播？ | 10+ |
| B2. 主动推理实现 | 感知+行动的统一框架如何工程化？ | 10+ |
| B3. 层级消息协议 | 层间信息传递的最优形式 | 5+ |

---

## B1：预测编码的深度学习化

### Rao-Ballard 协议——预测编码的通信标准

「Hierarchical Predictive Coding Models in a Deep-Learning Framework」(2020):

> Rao-Ballard 协议 = 预测编码层级间的通信语义
>
> **前馈**: 层i → 层i+1 传递**预测误差** ε_i = x_i − f_i(x_{i+1})
> **反馈**: 层i+1 → 层i 传递**预测** x̂_i = f_i(x_{i+1})
> **更新 (迭代)** : x_i ← x_i − η·∂ε_i/∂x_i

关键约束来自 Rao & Ballard (1999)：
- **统计独立性假设**: 各层误差假定为独立高斯
- **线性解码**: f_i 需保证可微，但推理层面不接受梯度消失
- **收敛时间**: 迭代至收敛后才更新权重——不同于 BP 的即时梯度

### 🎯 PCN: Deep Predictive Coding Network

「Deep Predictive Coding Network for Object Recognition」(2018):

> 前馈传递误差↑ + 反馈传递预测↓ + 递归循环直到收敛
> CIFAR-10/100 上始终优于前馈对照模型
> 递归越多次 → 性能越高（展开为时间上越来越深的模型）

**三大连接类型**:
```
↓ feedback: L_i+1 → L_i  (预测低层表征)
↑ feedforward: L_i → L_i+1  (传递预测误差)
↺ recurrent: L_i → L_i  (循环更新直到稳定)
```

### 🔬 µPC: 首次实现 100+ 层 PCN 稳定训练

「Towards Scaling Deep Neural Networks with Predictive Coding」(2025-10, PhD thesis):

> 发现: PC 学习动态 = 近似**信任域方法**，隐式使用二阶信息
> µPC 参数化：类似 µP，使深层 PCN 首次稳定训练
> 形式证明: PC 的有效学习景观比 MSE 景观**更良性**，更抗梯度消失
> 理论上 PC 可使用**任意高阶信息**

**关键突破**: PC 一直被批评"无法训练深层网络"——µPC 终结了这个论点。PC 不是只能做浅层——以前是参数化方式错误。

### 可微广义预测编码

「Differentiable Generalised Predictive Coding」(2021):

> 自动微分实现 PC——梯度基 PC 为每层局部优化**精度加权的预测误差**
> 层级预测编码**内容** + 动态预测编码**变化率及其高阶导数**
> 层级与动态预测**互相作用**于同一潜在状态——内容是"是什么"，动态是"在如何变"

**对白皮书消息协议的支撑**:
```
消息类型: 预测↓ = {content_prediction, dynamics_prediction}
消息类型: 误差↑ = {content_error, dynamics_error}
精度加权: error_effective = precision × raw_error
```

### 混合预测编码——快速与慢速推理的统一

「Hybrid Predictive Coding: Inferring, Fast and Slow」(2022):

> 前馈扫频 (feedforward sweep) = **摊销推理** (amortized inference)
> 递归处理 = **迭代推理** (iterative inference)
> **单一目标函数**统一两者——在熟悉的输入上用摊销快速获得信念，在异常输入上自适应切换到迭代推理
> 模型**内在感知不确定性**——高不确定性自动切换到迭代模式

**与白皮书"异常驱动计算"精确对应**:
白皮书公理：下层只上传压缩的意外信号
文献验证：当预测误差↓时 → 推理用快通道（摊销）；误差↑时 → 切换到慢通道（迭代）→ 仅对异常部分投入计算

---

## B2：主动推理的工程实现

### 奠基：「Deep Active Inference」(2017)

> 首次将 Friston 自由能原理与深度生成模型 + 进化策略结合
> VFE (变分自由能) 统一**感知、行动、学习**
> 深层循环网络实现变分密度的内部动态

**架构**:
```
观测 o_t → 编码器 → 潜在状态 s_t
s_t + 动作 a_t → 转移模型 → s_{t+1}
s_{t+1} → 解码器 → 预测 ô_{t+1}
目标: 最小化 VFE = 精度加权误差 + KL(s_t‖先验)
```

### Monte-Carlo 深度主动推理

「Deep active inference agents using Monte-Carlo methods」(2020):

> 四项技术创新:
> ① **MC 树搜索**选择自由能最优策略
> ② **前馈"习惯"网络**近似最优策略分布
> ③ **MC Dropout**预测未来参数信念更新
> ④ **状态转移精度优化**（= 高级注意力形式）

实验发现: 主动推理 agent 自动产生**解缠表征**——有利于状态转移建模。在 Animal-AI 中**即使视觉输入暂停**仍能模拟未来状态转移，即"想象"。

### Active Inference Tree Search (AcT)

「Active Inference Tree Search in Large POMDPs」(2021):

> 双向增强: 树搜索让主动推理可处理大规模问题；主动推理为搜索提供**原则性的探索-利用方案**
> 成功导航: 采样法困难的二叉树 + 需要自适应探索的问题 + **RockSample POMDP** 再现 SOTA 解
> 可模拟海马-PFC 神经生理响应

### Object-based Active Inference (OBAI)

「Object-based active inference」(2022):

> 将世界建模为**独立对象**——每个对象有独立的变分信念
> 选择性注意力**路由输入到对应对象槽**
> 学到的动态以对象为粒度——动作扰动后**正确分割视频中的对象**

**对白皮书"对象场"的支撑**: L3 的φ_k(x,t) 为每对象分配独立神经场 + 选择性注意力路由 → OBAI 提供了主动推理版本的对应实现。

### Action-Unaware vs Action-Aware

「Active inference for action-unaware agents」(2025-08):

> Action-aware: agent 知道自己的动作（有"传出副本"）→ 更好规划
> Action-unaware: agent 必须**推断自己做了什么**——从观测中反推
> **惊人结果**: action-unaware 在严重劣势下仍能追上 action-aware 的性能

**对白皮书的启示**: L3 的运动门控通道是一个选择——agent 可以直接接入传出副本（action-aware）以简化推理。不做 action-unaware 的自虐式设计。

---

## B3：层级消息协议——最优信息传递

### 消息协议的形式化

从上述文献中提取白皮书层级协议的公理基础：

```
消息 = {type, payload, precision}

消息类型:
├── PRED↓ (顶层→底层) : 对下层状态的先验预测
├── ERROR↑ (底层→顶层) : 精度加权的预测误差
├── GLOBAL⇄ (全局态广播) : L5→所有层的元参数
└── PRIOR↓ (先验注入) : L4→L3 的因果约束

precision 域: ℝ⁺ | ∞ = 确定性先验
```

### 精度加权——不是选加，是必须

「Differentiable Generalised Predictive Coding」(2021):
> 精度 = 置信度的逆——低精度的误差被降权
> 这意味着数据噪声**自动被过滤**，无需额外去噪模块

「Hybrid Predictive Coding」(2022):
> 精度还控制**推理模式切换**——高误差+低精度 = 切换到迭代
> 模型**自动感知自己的不确定性**——精度是内在量，来自模型自身

### 消息的经济性——只传"意外"

```
传统ML: L_i → L_{i+1} (全部激活值)
PC: L_i → L_{i+1} (压缩后的误差信号)

经济性 = 1 − |ε| / |activation|

高可预测时，误差接近 0 → 信息量接近 0 → 不消耗通信带宽
这正是 Rao-Ballard 协议的统计假设——正常时几乎无信号流动
```

### 消息协议的扩展——非线性的层级拓扑

「Generalised PC」(2021) 的开放问题:
> 线性层级假设 → 更灵活的**图结构**，具有涌现性质
> Markov Blanket 刻画模块性——模块间的信息通过界面流

支持白皮书中的**多智能体延伸**: 消息协议不仅用于层间，也用于 agent 间——Markov Blanket 定义了 agent 的自主性接口。

---

## 关键文献优先级

### 🟢 Tier 1: 必须引用

1. **「Hierarchical Predictive Coding Models in Deep Learning」**(2020) — RB 协议的完整分析
2. **「Deep Predictive Coding Network」**(2018) — PCN 三连接类型 + 优于前馈
3. **「Towards Scaling PC: µPC」**(2025-10) — PC 深层训练的理论突破
4. **「Hybrid Predictive Coding: Fast & Slow」**(2022) — 摊销+迭代的统一
5. **「Deep Active Inference」**(2017) — AIF+DL 的奠基

### 🟡 Tier 2: 重要支撑

6. **「MC Deep Active Inference」**(2020) — 四项创新技术
7. **「Active Inference Tree Search」**(2021) — AIF+树搜索
8. **「Differentiable Generalised PC」**(2021) — 层级+动态预测
9. **「Object-based Active Inference」**(2022) — 对象槽+注意力
10. **「Deep Variational RL for POMDPs」**(2018) — DVRL 潜在状态聚合

### 🔵 Tier 3: 拓展

11. 「Action-Unaware vs Action-Aware」(2025-08)
12. 「Dynamic Early Exiting PCN」(2023)
13. 「Variational Message Passing」(2018)

---

## 与本框架白皮书的映射

| 白皮书主张 | 文献支撑 | 强度 |
|------------|----------|------|
| 预测编码可替代 BP | µPC(2025): 100+层PCN训练; PC=近信任域 | ⭐⭐⭐ |
| 误差↑ / 预测↓ 协议 | RB 协议 (Rao-Ballard 1999) → 多篇验证 | ⭐⭐⭐ |
| 精度加权 = 信噪分离 | Generalised PC (2021): 精度=置信逆 | ⭐⭐⭐ |
| 混合推理 (快+慢) | Hybrid PC (2022): 单目标统一摊销+迭代 | ⭐⭐⭐ |
| 主动推理统一感知/行动 | Deep AIF (2017) + MC AIF (2020) | ⭐⭐⭐ |
| 对象级神经场 + 注意力 | OBAI (2022): 对象槽+选择性注意力 | ⭐⭐ |
| 传出副本信号 (可选) | Action-Unaware (2025): 副本可简化推理 | ⭐⭐ |
| 层级非线性拓扑 | Generalised PC (2021) 开放问题 | ⭐ 指向方向 |

---

## 综述的三条核心论据

### 论据 1: µPC 为预测编码正名——PC≠反向传播，但可训练深层

µPC PhD thesis (2025) 的三重论证——(i) PC = 隐式二阶信任域; (ii) µPC 参数化使 100+ 层稳定训练; (iii) PC 有效景观比 MSE 景观更良性——摧毁了"PC 只能做浅层"的长期批评。白皮书选择 PC 作为学习机制是有工程基础的——不是纯理论空想。

### 论据 2: 混合推理是白皮书消息协议的天然伸展

Hybrid PC (2022) 证明单一目标函数就能自适应切换"快速前馈"与"慢速迭代"推理。切换条件直接由**该层的预测误差**大小决定。白皮书的"只传意外信号"方案在文献中以数学优化（不是手工规则）实现。

### 论据 3: 主动推理已有工程化路线图

Deep AIF (2017 → 2020 → 2021 → 2022) 展示了完整演进：从低维玩具 → MC 混合 → 树搜索大规模 POMDP → 对象级推理。白皮书的 L2→L3→L4→L5 可与这演化路线**对齐实现**——每个版本对应一个层级的成熟度。
