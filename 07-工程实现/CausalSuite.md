# CausalSuite v1.0 — L2 具身交互接口环境族

> **版本**: v1.0 · **日期**: 2026-05-12
> **定位**: L2 具身交互接口（原"感知-行动层"）的验证沙盒集合
> **背景**: 三专家（Qwen/ChatGPT/DeepSeek）独立指出"L2 过于单薄（仅 CausalPong）"，本文档作为 P1 交付物固化 CausalSuite 编目

---

## 一、设计原则

1. **渐进复杂度**: 从 2 球到 N 体，因果结构复杂度递增
2. **GT 可验证**: 每个环境都有 `make_*_ground_truth()` 生成精确的因果图
3. **接口统一**: 所有环境都实现 `generate_dataset()`，返回统一格式的 episode dict
4. **CAL 兼容**: episode dict 字段统一，CAL 可直接消费

---

## 二、环境一览表

| 环境 | 文件 | 物体数 | 因果边 | DAG | 复杂度 | 用途 |
|------|------|--------|--------|-----|--------|------|
| **CausalPong** | `causal_pong.py` | 2 | 1 (collision→ball_b_vel) | ✅ | ⭐ | M0-M4 基础验证 |
| **CausalBounce** | `causal_bounce.py` | 3 | 2 (ball_i→ball_j collision) | ✅ | ⭐⭐ | 多物体碰撞验证 |
| **CausalGravity** | `causal_gravity.py` | 3 | 30 (双向引力耦合) | ❌ | ⭐⭐⭐ | 非因果闭环反馈 |
| **CausalChaos** | `causal_chaos.py` | 可变 | 可变 | 可变 | ⭐⭐⭐⭐ | 混沌/高敏感度验证 |

---

## 三、统一接口契约

所有 `generate_dataset()` 函数返回 episode list，每个 episode 是 dict：

```python
episode = {
    # ── 必选字段（所有环境统一）───
    "frames":      Tensor[T, 1, H, W],     #  occupancy rendering
    "collisions": Tensor[T],               #  binary collision flags
    "actions":    Tensor[T, ...],          #  agent actions

    # ── 位置字段（CAL 消费）───
    "ball_*_x":   Tensor[T],              #  per-object X positions
    "ball_*_y":   Tensor[T],              #  per-object Y positions
    #  * 匹配各环境的物体命名（pong: a/b, bounce: 0/1/2, ...）

    # ── 元数据───
    "episode_id": int,
    "config":      Dict,                   #  environment config snapshot
    "n_objects":  int,
    "n_frames":   int,
}
```

CAL 的 `extract_from_episode()` 依赖的字段：
- `ball_*_x`, `ball_*_y` → 运动学事件（EventType.KINEMATIC）
- `collisions` + 位置 → 碰撞事件（EventType.COLLISION）
- `actions` → 策略事件（EventType.POLICY）

---

## 四、各环境详细说明

### 4.1 CausalPong（2 球·最小验证）

```python
# 文件: src/data/causal_pong.py
# GT:  make_pong_ground_truth(n_balls=2)

## 因果图 (8 条边, DAG=True)
variables = ['ball0_x','ball0_y','ball0_vx','ball0_vy',
              'ball1_x','ball1_y','ball1_vx','ball1_vy',
              'collision','policy']

## 物理边（6 条·运动学）
ball0_x → ball0_vx   (vx = dx/dt)
ball0_y → ball0_vy
ball1_x → ball1_vx
ball1_y → ball1_vy
# + 速度→位置（时间自回归，由 Granger 处理，不含于 DAG）

## 碰撞边（2 条）
collision → ball1_vx   (B 的 vx 在碰撞后改变)
collision → ball1_vy   (B 的 vy 在碰撞后改变)
# 注意: ball_a_bounce=False 时 collision→ball0_v* 不存在

## 策略边（0 条，chase policy 是外部干预）
# policy → ball0_v* 在观测数据中可被 Granger 检测，但非物理因果
```

**已知限制**（E4.1 诊断）：
- Ball A 的 chase policy 与 collision→ball0_v 叠加 → 混淆，CI 检验无法分离
- 碰撞率稀疏（~3%）→ collision 边 CI 检验 power 不足
- **结论**: 物理 GT ≠ 观测可恢复图；需要 `do(action=hold)` 干预

---

### 4.2 CausalBounce（3 球·多物体碰撞）

```python
# 文件: src/data/causal_bounce.py
# GT:  make_bounce_ground_truth(n_balls=3)

## 因果图 (18 条边, DAG=True)
variables = ['ball0_x','ball0_y','ball0_vx','ball0_vy',
              'ball1_x','ball1_y','ball1_vx','ball1_vy',
              'ball2_x','ball2_y','ball2_vx','ball2_vy',
              'collision_0_1','collision_0_2','collision_1_2']

## 碰撞边（6 条）
collision_0_1 → ball1_vx, ball1_vy
collision_0_2 → ball2_vx, ball2_vy
collision_1_2 → ball2_vx, ball2_vy
# + 对称边（如果 ball_i_bounce=True）
```

**用途**: 验证公理 6（组合完备）— 多物体场景 Slot Attention / Gumbel-Softmax 表现

---

### 4.3 CausalGravity（3 体·引力耦合）

```python
# 文件: src/data/causal_gravity.py
# GT:  make_gravity_ground_truth(n_bodies=3)

## 因果图 (30 条边, DAG=False ❌)
variables = ['body0_x','body0_y','body0_vx','body0_vy',
              'body1_x','body1_y','body1_vx','body1_vy',
              'body2_x','body2_y','body2_vx','body2_vy']

## 引力边（双向）
body0_x → body1_x, body0_x → body2_x, ...
# N 体引力本质是双向反馈，不构成 DAG
# Granger 检验会产出大量 bidir 边（符合预期）
```

**用途**: 验证 PC_Algo 在闭环反馈系统中的表现；测试 CAL 处理 bidir 边的能力

---

### 4.4 CausalChaos（混沌·高敏感度）

```python
# 文件: src/data/causal_chaos.py
# GT:  make_chaos_ground_truth(...)

## 因果图（可变，依赖 chaotic parameter G）
# 混沌系统的因果结构对初始条件极度敏感
# 用于测试因果发现的鲁棒性
```

**用途**: E4.3 CausalScore 在高敏感度系统中的 calibration；M6 PRIOR 协议在混沌环境中的表现

---

## 五、CAL 环境适配状态

| 环境 | CAL 支持 | 备注 |
|------|-----------|------|
| CausalPong | ✅ 完整 | `extract_from_episode()` 直接使用 |
| CausalBounce | ⚠️ 需扩展 | `ball_*_x/y` 命名需泛化（当前写死 a/b） |
| CausalGravity | ⚠️ 需扩展 | 需处理 bidir 边标注 |
| CausalChaos | ⏳ 待实现 | CAL 事件类型需扩展 |

---

## 六、扩展路线图

### Phase 2a: CausalBounce 完整集成（预计 1-2h）
1. 泛化 CAL 的 `extract_from_episode()` 支持 `n_balls` 可变
2. 更新 `ball_*_x/y` 字段名解析逻辑
3. 验证 `event_stream_to_graph()` 产出 18 条边

### Phase 2b: CausalGravity bidir 边处理（预计 1h）
1. CAL 事件增加 `is_directed` 字段
2. `CausalGraph` 支持 bidir 边渲染
3. PC_Algo 结果评估指标更新（允许 bidir 匹配）

### Phase 2c: 高维连续动作（未来）
- CausalPong → 连续控制（steer, accelerate）
- 测试 L4 因果发现在连续动作空间的效率
- 需要 RL 驱动的因果探索（ChatGPT 建议）

---

## 七、与 L3/L4 的集成状态

```
L2 CausalSuite ──episode dict──▶ CAL ──CausalEventStream──▶ L4 PC_Algo
                            │
                            └──▶ L3 SIREN 训练（M1/M2/M3）
```

- ✅ CausalPong 已完整集成（M0-M4, E4.1-E4.5）
- ⚠️ CausalBounce/Gravity 仅 GT 生成器，CAL 适配待完成
- ⏳ CausalChaos 完整集成待 Phase 2c

---

## 八、测试覆盖

| 环境 | 冒烟测试 | E4.x 实验 | CAL 提取验证 |
|------|----------|-----------|----------------|
| CausalPong | ✅ | E4.1-E4.5 | ✅ (冒烟测试通过) |
| CausalBounce | ⚠️ 部分 | — | ⏳ |
| CausalGravity | ⚠️ 部分 | — | ⏳ |
| CausalChaos | ❌ | — | ❌ |

---

*本文档由三专家交叉验证驱动，作为 L2 定位修正（P1）的核心交付物。*
