# APL 系统总览（简化版本之后的最终说明）

> 这是给**第一次接触本仓库**或者**只想搞清来龙去脉**的人看的一篇文档。
> 它不是 API 参考，而是一份"从空中俯瞰整个系统"的解释。
> 配套文档：[GLOSSARY.md](GLOSSARY.md)（术语速查）、[../README.md](../README.md)（命令清单 + 22 模板表）。

---

## 1. 项目到底在做什么

**输入**：一栋已经做过 3D 重建的房子（3D Gaussian Splatting 点云 + 物体 AABB 标注 + 房间多边形）。

**输出**：成千上万条空间推理题，**每条题不是单张图，而是一整段动作轨迹**：

```
(初始视角, 题目, 选项,   ← 题面：从这一帧开始问
 expert_trajectory,     ← 标准答案：从初始视角出发，怎么走到能回答的位置
 最终视角, 答案,         ← 落脚点：能直接看见证据并作答
 coverage, score)        ← 这条轨迹的客观质量
```

为什么这样设计？因为目标模型是个 **3D 室内主动感知智能体**，不是普通 VQA 模型。普通 VQA 给一张图问问题；本系统问的是"你不在合适的位置，去走到合适的位置，再回答"。我们要训练它**主动选视角**的能力，因此题目自带一段"动起来"的过程。

一句话总结：**本仓库是一个把 3D 场景自动转写成"带专家轨迹的空间推理题"的数据工厂**。

---

## 2. 一道题在系统里的真实结构

打开 `out/batch/T05.jsonl` 任意一行（已格式化）：

```jsonc
{
  "task_id":      "T05_active_b774991f",
  "template_id":  "T05",         // 这道题属于哪一类
  "subclass":     "C1.2",         // 子类标签，仅做归档
  "question":     "Which is nearer to the niche: the towel or the flowers?",
  "choices":      ["towel", "flowers"],
  "answer":       "towel",

  "init_view":    {"position":[...], "target":[...]},   // 起点，故意问不出
  "target_view":  {"position":[...], "target":[...]},   // 终点，能看见证据
  "expert_trajectory": [ViewState, ViewState, ...],     // 中间的每一帧位姿
  "action_descriptions": ["move_right", "turn_right", ..., "stop"],

  "quality_spec": { "evidence_slots":[...], "gamma":0.95, "min_coverage_for_credit":1.0 },
  "coverage":     1.000,
  "score":        0.774,
  "num_steps":    6
}
```

读出来要知道的几个事实：

- **`init_view` 是故意挑出来回答不了的那一帧**——这是系统的核心强制条件。
- **`target_view` 是用一组叫"evidence slot"的几何条件挑出来的合格帧**。
- **`expert_trajectory` 是 beam search 找到的从 init 到 target 的最短动作序列**。
- **`coverage` 只看最后一帧**：路过合格视角不算分。
- **`score = 1[答对] · min(1, Cov/min_cov) · γ^T`**：答对、证据够、步数少，三者乘起来。

---

## 3. 系统的几何原语层（核心积木）

在讲流程之前，先把所有 "工具" 列清楚——后面 7 步只是把它们组合起来用。

### 3.1 动作空间（10 个 primitives）

源码：[action_space/action_primitives.py](../action_space/action_primitives.py)

| 原语 | 默认参数 | 物理含义 |
|---|---|---|
| `MOVE_FORWARD` / `MOVE_BACKWARD` | 0.5 m | 沿/反 camera-forward 平移 |
| `MOVE_LEFT` / `MOVE_RIGHT` | 0.5 m | 沿水平左/右侧向平移（strafe）|
| `TURN_LEFT` / `TURN_RIGHT` | 45° | yaw 旋转，视点不变 |
| `LOOK_UP` / `LOOK_DOWN` | 15° | pitch 旋转 |
| `LOOK_FORWARD` | — | 把 pitch 复位到水平 |
| `STOP` | — | 提交。**只有这一帧的视角才计入打分**（v3 反作弊）|

`ActionConfig` 还可以给每个原语配置 `*_variants`（如 `move_variants=[0.3, 0.5, 1.0]`）让 beam search 试不同步幅。`max_sequence_length` 默认 10。

**关键约束**：状态空间是连续的 5 维（`x,y,z,yaw,pitch`），但动作是离散的——这让 beam search 可解，又保留了精细调整的可能。

### 3.2 几何 predicate（12 个，单帧判定）

源码：[evaluation/predicates.py](../evaluation/predicates.py)。每一个 predicate 接收 `(cam_pos, cam_target, hfov, scene_ctx, **kwargs)`，返回 `bool`——**这是系统判定 "看见了没" 的最小原子**。

| 名称（snake / Pascal）| 关键参数 | 通过条件 |
|---|---|---|
| `is_visible` / `Visible` | `obj`, `min_corners=4`, `max_occ=0.30` | AABB 至少 4 个角在视锥内 **且** 被遮挡比例 ≤ 30% |
| `pair_visible` / `PairVisible` | `obj_a`, `obj_b` | 两个对象同时通过 `Visible`（默认放宽：3 角 + 40% 遮挡）|
| `is_centered` / `Centered` | `obj`, `theta_deg=25` | 视线与 "camera→object 中心" 的夹角 ≤ θ |
| `distance_band` / `DistanceBand` | `obj`, `d_min`, `d_max` | 相机到 obj 中心距离 ∈ \[d_min, d_max\]（米）|
| `scale_band` / `ScaleBand` | `obj`, `s_min=0.05`, `s_max=0.70` | 投影 AABB 占图像面积的比例 ∈ \[s_min, s_max\] |
| `bearing_within` / `BearingWithin` | `obj` 或 `point`, `theta_deg` | 目标 bearing（带正负号 yaw）在 ±θ° 窗内 |
| `aspect_exposed` / `AspectExposed` | `obj`, `min_ratio=0.30` | 投影 AABB 短边/长边 ≥ min_ratio（防极扁视角）|
| `in_room` / `InRoom` | `room_id` | 相机位置落在指定房间多边形内 |
| `not_blocked_by` / `NotBlockedBy` | `obj`, `blocker?` | 主要遮挡者不是 blocker（或总遮挡 < 50%）|
| `orthogonal_to_plane` / `OrthogonalToPlane` | `plane`, `theta_max_deg=15` | camera-forward 与平面法向夹角 ≤ θ（即视线 "撞" 该面）|
| `separation_on_image` / `SeparationOnImage` | `obj_a`, `obj_b`, `min_separation_px=40` | 投影中心像素距离 ≥ 阈值（区分粘连）|
| `side_view` / `SideView` | `obj`, `side`, `min_angle_deg=30`, `max_angle_deg=80` | 视线与该 face 法向夹角落在 \[30°, 80°\]（拍侧脸而非正脸）|

背后真正干活的是 `SceneContext` 上的几个查询函数：`visible_corner_mask` / `occlusion_fraction` / `project_aabb_corners` / `primary_occluder` / `get_object_centre` / `get_plane_normal`——所有 predicate 的几何计算最终落在这一层。

这 12 个 predicate **是组合式的**：T05 只用 `PairVisible` + `Centered`；T11 用 `Visible` + `SeparationOnImage`；T29 用 `SideView` + `Centered`。组合三四个 predicate 就能精确刻画绝大多数空间证据。

### 3.3 Region generator（16 个，候选位姿采样）

源码：[evaluation/region_generators.py](../evaluation/region_generators.py)。Region generator 的作用是**把 "哪里可以满足 predicate" 的搜索空间从整间屋子缩到几十个候选位姿**——不然 beam search 起步都出不来。

每个 generator 接收 `(scene_ctx, rng, **kwargs)`，返回一组 `(pos, target, height, yaw_deg)` 候选。

| 名称 | 几何意图 |
|---|---|
| `around_object` | 在某 obj 周围一圈采样，朝向它（特写镜头）|
| `around_pair` | 在 A、B 中点周围采样并朝向中点（合影 framing）|
| `equidistant_to_pair` | 在 A、B 中垂面附近采样（公平比较距离）|
| `view_of_triple` | 三对象同框（T08 几何构型题）|
| `view_with_separation` | 强制两对象在像素上分开（T11 单/多歧义）|
| `view_breaking_projection` | 主动选 "投影会骗人" 的角度（T04 真实大小）|
| `behind_front_object` | 走到前景对象的背后侧（T13/T17 露出 bg）|
| `behind_object_face` | 绕到 obj 某 face 的背面（T14）|
| `around_occluder_back` / `around_occluder_to_expose` | 绕 occluder 暴露被挡区域 |
| `orthogonal_to_portal` / `orthogonal_to_segment` | 站在与门/线段垂直的位置（T24/T33）|
| `near_portal_in_room_a` / `near_portal_facing` | 在 room A 一侧靠近 portal |
| `bearing_sector_in_room` | 在房间内某 bearing 扇区采样（T20）|
| `inside_room` | 房间内任意位置（兜底，最弱约束）|

**怎么挑 generator**：YAML 里直接写名字。看 [task_generation/templates/p0/T05_*.yaml](../task_generation/templates/p0/T05_actual_distance_comparison.yaml) 的 `evidence_slots[*].region_generator` 字段。

---

## 4. 完整生成管线（7 步）

每次调用 `generate_for_template(template_id="T05", n=5)` 就会重复跑这 7 步 5 遍。

### Step 1 — 加载模板

```python
spec = load_template("T05")      # 读 task_generation/templates/p0/T05_*.yaml
```

YAML 描述这一类题：题面文字、答案怎么算、需要看到什么、判分参数。**YAML 是这套系统唯一的题目定义来源**，Python 那边没有第二份。

最新的 YAML 推荐写法（简化后的统一 schema）：

```yaml
template_id: T05
question_templates: ["Is the {a} closer to {b} or {c}?"]
checks:                                  # ← 新统一写法
  - {when: scene,  name: min_objects, args: 3}
  - {when: init,   name: max_both_pairs_visible_at_init, args: false}
  - {when: target, slot: AB_view,
     name: PairVisible, args: {obj_a: "{{a}}", obj_b: "{{b}}"}}
```

`checks:` 在加载时会被自动展开成旧字段（`scene_requirements`/`trigger`/`evidence_slots[*].predicates`）；19 个老 YAML 依旧能加载——两种写法长期共存。

### Step 2 — 实例化（占位符 → 真物体 ID）

```python
ti = INSTANTIATORS["T05"](spec, scene_ctx, rng)
# ti = {"subject_id": 17, "ref_b_id": 4, "ref_c_id": 9,
#       "subject_label": "niche", "ref_b_label": "towel", ...}
```

YAML 里写的是 `{subject_id}`、`{ref_b_id}`，这一步把它们换成场景里真正存在的物体 ID。`_instantiate_T05` 在 `task_generation/apl_tasks/template_active_generator.py` 里——一个模板对应一个函数。

如果场景里凑不出三个满足 trigger 条件（距离比 ≥1.3 等）的物体，函数抛 `SceneRequirementUnmet`，这条尝试作废。**这不是 bug，是设计**：场景不够丰富就出不了这道题。

> 简化项 Item 7 之后：可以用 `get_template_handler("T05")` 一次拿到 `(spec, instantiator)`，不用再分两步查。

### Step 3 — 证明可行（target view 存在）

对每个 `evidence_slot` 走 "采样–判定" 两步：

1. 用该 slot 的 `region_generator` 撒 N（默认 ~30）个候选位姿。
2. 对每个候选，逐个跑 slot 里写的 `predicates`，**全部通过** 即合格。
3. 至少一个候选合格 → 该 slot 可达；**所有 slot 都至少有一个合格候选 → 这道题可解**。

几何上的 "看见了" 是 `fraction_passed = 通过的 predicate 数 / 总 predicate 数`；slot 默认 `threshold=1.0`（必须全过）。tier-2 的整段轨迹判定（`tier2_override`）是少数模板用的——比如 "轨迹中确实绕到了对象背面" 这种**需要看序列**而非单帧的判据。

### Step 4 — 选 init view（故意问不出）

这步**最容易出 bug 的题**——所以系统专门为它设计了一层 `init_validators` 字段分发表。

```python
passes, pref, breakdown = score_init_view(spec, ti, cp, ct, hfov, ctx)
```

#### 4.1 init 筛选的两条原则

1. **硬约束（必须满足，否则丢弃）**：
   - 至少一个 slot 的 predicate **不满足** → 这道题从这一帧问出来确实答不了；
   - YAML `trigger:` 里所有声明的字段都通过对应的 validator（任何一个回 `-inf` 都判负）。
2. **软偏好（在多个合法候选里挑分高的）**：
   - 不同 "失败方式" 不一样好。比如 T20（局部方位搜索）要求 target 看不见——target 在 90° 侧方比 0° 正前被挡更典型，更适合训练 "转头去看"。
   - 系统采若干合法候选，按 `preference` 取最高那一个当 init view。

#### 4.2 验证器评分档（在 `init_validators.py` 内部使用，对外只暴露 bool+float）

| 档位 | preference | 含义 |
|---|---|---|
| 理想 | 1.0 | 题目有意义、非平凡、视觉锚点充足 |
| 可接受 | 0.5–0.7 | 非平凡但锚定一般 |
| 兜底 | 0.1–0.3 | 非平凡但 agent 几乎没上下文（比如对着墙）|
| 中性 | 0.0 | 字段存在但没区分度 |
| 硬拒 | -inf | 模板的严格不变量被破坏 |

#### 4.3 全部 20 个已注册 validator（按 YAML 字段名分发）

源码：[task_generation/apl_tasks/init_validators.py](../task_generation/apl_tasks/init_validators.py)。`spec.trigger` 字典的每一个 key 都会被尝试在这张表里查；查不到的字段一律忽略（debug 一行）；`_NON_INIT_TRIGGER_FIELDS` 列表里的字段是 "实例化时用的"（如 `min_dist_ratio`），不当 init 约束。

| YAML 字段 | 干什么 | 用它的模板 |
|---|---|---|
| `target_invisible_at_init` | 强制 target 在初始帧看不见；偏好 "在侧后方" | T17/T18/T20/T23/T26/T27 等 |
| `max_both_pairs_visible_at_init` | 不让 (A,ref) 与 (B,ref) 同时全可见（否则秒答）| T05/T21 |
| `max_init_separation_frac` | A、B 在像素上不能拉开太远（强迫先聚再判）| T05/T21 |
| `min_init_occlusion_fraction` | target 至少被挡 X%（T13/T17 的 "被挡" 题面）| T13/T17/T19 |
| `min_target_occlusion_at_init` | 同上，更严格的阈值 | T18 |
| `max_init_back_angle_deg` | target 不在 "完全 180° 背后"，避免无脑掉头 | T20 |
| `max_init_lateral_angle_deg` | target 不能正好 90° 侧方（防过分简单）| 部分 T2x |
| `label_side_visible_at_init` | 物体的标签面**不能**已经看到（T15）| T15 |
| `only_one_face_visible_at_init` | T14：只看到一面，必须绕过去 | T14 |
| `portal_invisible_at_init` | 通往 target_room 的门初始不可见 | T23/T24 |
| `min_init_occluded_count` | 同类对象至少有 N 个被挡（T26 计数题）| T26 |
| `min_zones_invisible_at_init` | 至少 K 个 zone 的内容不可见（T27 多区计数）| T27 |
| `init_apparent_size_ratio_min/max` | 投影面积比落在带宽内（T04 制造 "近大远小" 错觉）| T04 |
| `min_projection_ambiguity_deg` | 视线被某轴 "压扁" 的程度（T11 单多歧义）| T11 |
| `init_ordering_ambiguous` | 三对象在像素上的左右序与真实序不一致（T08）| T08 |
| `target_invisible_from_room_a` | 站 room A 看不到 target（跨房间题）| T23 |
| `path_portals_not_all_visible_at_init` | 沿路径的 portal 不能初始全可见（T32）| T32 |
| `passage_not_orthogonally_visible_at_init` | 通道初始不能正向对着（T33）| T33 |

> 想加新模板的新 trigger 字段？直接 `@register_init_validator("my_field")` 注册一个返回 `[0,1]∪{-inf}` 的函数，多个模板都可以共用。

#### 4.4 `init_view` 到底是怎么选出来的（新手版）

源码入口在 [task_generation/apl_tasks/template_active_generator.py](../task_generation/apl_tasks/template_active_generator.py) 的 `_sample_failing_init`。它不是"随机挑一帧"，而是一个**先过滤再打分**的过程：

1. 先采候选：
    - 在所有房间多边形里先采 `n_pos=64` 个位置；
    - 每个位置再试 `yaws_per_pos=4` 个随机朝向；
    - 总计约 256 个候选视角。
2. 对每个候选算三件事：
    - `n_strict`：有多少 slot 在这个视角下是"严格满足"（`fraction_passed >= threshold`）；
    - `n_relaxed`：有多少 slot 是"接近满足"（阈值乘 0.5），代表看到了部分线索；
    - `validator_score`：把 YAML `trigger` 字段逐个送进 `init_validators`，累加偏好分。
3. 先做硬过滤（保证题目成立）：
    - 如果 `n_strict == n_total`，直接丢弃（说明这一帧已经能答题，不是合格 init）；
    - 如果任一 validator 返回 `-inf`，直接丢弃（破坏模板硬约束）。
4. 对通过硬过滤的候选打总分：

$$
	ext{total} = 1.0 + 0.5\cdot\frac{n_{\text{relaxed}}}{n_{\text{total}}} + \text{validator\_score}
$$

最后选分数最高的候选作为 `init_view`；如果没有候选通过，则本次实例化失败并重抽。

这套流程保证了两件关键事实：
- **一定答不出来**：因为 `n_strict < n_total` 是硬约束；
- **又不是完全瞎走**：`n_relaxed` 和 validator 偏好会优先保留"有线索但不充分"的起点，利于训练主动探索。

### Step 5 — 跑专家搜索

```python
expert_traj, final_view = find_expert_trajectory(
    init_view, slots, scene_ctx, max_steps=12, gamma=0.95)
```

源码：[evaluation/expert.py](../evaluation/expert.py) + [evaluation/potential.py](../evaluation/potential.py)。

#### 势函数 Φ

$$
\Phi(s) \;=\; \frac{1}{|S|}\sum_{\text{slot}\in S}\frac{\#\{\text{predicate of slot passes at }s\}}{|\text{predicates of slot}|}
$$

Φ ∈ \[0, 1\]；越接近 1 表示越接近 "全部 slot 全部 predicate 通过"。**注意**：tier-2 整轨迹判据在 Φ 里近似为只看预测帧（per-state 没法整体判），所以 Φ 是搜索的 *启发*，不是终判。

#### Beam search 主循环

- 状态 = ViewState `(x, y, z, yaw, pitch)`；动作 = 上面的 10 个原语。
- 塑形奖励 $r_t = \Phi(s_{t+1}) - \Phi(s_t) \in [-1, 1]$。
- beam 按累积回报排序保留 top-K。
- 终止条件：某条路径执行 `SUBMIT` 且所有 slot 在该帧 `fraction_passed ≥ threshold`。
- 剪枝键 `(round(x/0.25 m), round(y/0.25 m), round(yaw/22.5°))` 去重，防状态爆炸。

**理论保证**：γ = 0.95 时每步隐式成本 ≈ 5%。所以塑形奖励小于 `1 − γ = 0.05` 的步是 "得不偿失" 的——这给搜索提供了 "何时该 SUBMIT" 的客观尺度。

### Step 6 — 打分（视角评分原则）

```python
coverage = compute_coverage(template, trajectory, ti, ctx, submit_only=True)
score    = episode_score(template, trajectory, predicted, gt, ti, ctx)
```

源码：[evaluation/coverage.py](../evaluation/coverage.py) + [evaluation/scorer.py](../evaluation/scorer.py)。

#### 6.1 三个评分原则（按重要性排序）

1. **答对优先（hard gate）**：`predicted ≠ gt` → score = 0，其他什么都不看。
2. **只看 SUBMIT 那一帧（v3 anti-cheat）**：`compute_coverage(..., submit_only=True)`——路过合格视角不算。这是**整套系统能训出 "主动停下" 行为的关键**。
3. **步数惩罚不可关**：每多走一步乘一次 γ。

#### 6.2 公式（写明白）

$$
\text{Score}\;=\; \mathbb{1}[\hat{a}=a^*]
\;\cdot\;\min\!\Big(1,\;\tfrac{\text{Coverage}}{\text{min\_coverage\_for\_credit}}\Big)
\;\cdot\;\gamma^{T}
$$

其中：
- $\text{Coverage} = \dfrac{1}{|S|}\sum_{\text{slot}\in S}\mathbb{1}\!\left[\text{fraction\_passed at SUBMIT view} \geq \text{slot.threshold}\right]$
- $T = \text{len(trajectory)} - 1$（步数）。
- 默认 $\gamma = 0.95$，$\text{min\_coverage\_for\_credit} = 1.0$（必须 100% 才给分）。

#### 6.3 step-level 奖励（给 RL 用，可选）

`step_rewards()` 输出列表：$r_t = \Phi(s_{t+1}) - \Phi(s_t)$，最后一步加上 `episode_score`。**势能塑形**保证最优策略不变（Ng & Russell 1999），但给中间步提供稠密信号。

### Step 7 — 打包

填占位符进 `question_templates`，调 `build_choices` 生成 `choices`，按 `gt_from` 计算 `answer`，写一行 JSONL。

---

## 5. 为什么有这么多名词——每一个对应什么

| 名词 | 对应训练目标里的约束 |
|---|---|
| **template** (YAML) | "这一类题怎么生成" 的单一来源 |
| **template_id** (`T01..T33`) | 给这类题一个稳定的归档短码 |
| **subclass** (`C1.2`) | 进一步的标签，**仅用于统计**，对生成逻辑无影响 |
| **task_instance** | 把模板占位符填上具体 ID 后的字典 |
| **instantiator** | "怎么挑 ID" 的 Python 函数（每模板一个） |
| **scene_requirement** (when=scene) | 场景级别预筛：场景里没两个房间，就别想出跨房间题 |
| **trigger** (when=init) | 实例级别预筛：距离比太小、初始就能答出来——这道题不出 |
| **predicate** (when=target) | 单帧几何判定："这一帧到底算不算看见了" |
| **evidence_slot** | 一帧的需求清单：region + 一组 predicate |
| **region_generator** | 把搜索空间从整间屋子缩到几十个候选位姿（剪枝） |
| **init_view** | 故意挑出来"问不出来的起点"——强迫模型动起来 |
| **target_view** | 至少一帧能满足所有 slot 的视角，证明题目可解 |
| **expert_trajectory** | beam search 找到的"init→target 的最短路径"，做监督/RL 参照 |
| **$\Phi(s)$ potential** | 教搜索算法"哪一步靠近目标"的标量信号 |
| **coverage / score / $\gamma^T$** | 在最终帧客观判分；步数惩罚鼓励走最短路 |

没有一个名词是装饰。如果删掉其中某个，对应的设计目标就丢了：
- 删 `trigger` → 弱智题会被生成出来。
- 删 `region_generator` → 搜索空间太大跑不完。
- 删 `init_view` 强制失败 → 模型睁眼答题，学不到主动感知。
- 删 `coverage`（只判最后一帧） → 模型扫一圈作弊也得分。

---

## 6. 22 个模板速览（每道题考什么）

所有 22 个 P0 模板都遵循 **"初始看不见 → 走到位看见 → 提交"** 的主动感知范式。下表只列**这道题独有的认知考点**，详细 trigger / slot 见 [README §5](../README.md)。

| 模板 | 名字 | 子类 | 考的空间能力 |
|---|---|---|---|
| T01 | category_recognition | C1.1 | 远处看不清的物体走近识别**类别** |
| T04 | actual_size_comparison | C1.2 | 排除近大远小后比较**真实体积** |
| T05 | actual_distance_comparison | C1.2 | 同参考物到两目标的**真实距离**谁更近 |
| T06 | clearance_assessment | C1.2 | 物体能否穿过最窄通道 |
| T08 | configuration_judgment | C1.3 | 三对象**几何构型**（三角/共线）|
| T11 | single_multiple_split | C1.4 | 远看似一个其实是 **1 还是 2** |
| T13 | post_occlusion_continuation | C2.1 | 被挡延伸是**同物还是另物** |
| T14 | back_face_acquisition | C2.1 | 看物体**背面**（需 `front_normal` 标注）|
| T15 | label_side | C2.1 | 标签朝向哪面（需 `label_side` 标注）|
| T16 | front_back_difference | C2.1 | 正反面差异 |
| T17 | post_occlusion_existence | C2.2 | 被挡的位置**后面有没有东西** |
| T18 | post_occlusion_category | C2.2 | 被挡的是什么类别 |
| T19 | post_occlusion_completeness | C2.2 | 被挡对象**是否完好**（当前实现默认 GT=complete，可不做人工 completeness 标注）|
| T20 | local_bearing_search | C2.3 | 周围 360° **方位搜索**到某物 |
| T21 | local_nearest_target | C2.3 | A、B 哪个**离参考物更近** |
| T23 | cross_room_existence | C3.1 | 另一房间里**有没有 X** |
| T24 | portal_direction | C3.1 | 通往房间 X 的**门在哪个方向** |
| T26 | occluded_counting | C3.2 | 含被遮挡在内**数总数** |
| T27 | zone_counting | C3.2 | 某/多个房间内 X **共几个** |
| T29 | contact_relationship | C3.3 | A 放在 B **上面/旁边/分离** |
| T32 | connectivity_judgment | C3.4 | 两房间**直接/经某房间**连通 |
| T33 | passage_passability | C3.4 | 通道**够不够某尺寸通过** |

**子类组别**：C1=单物体几何错觉，C2=遮挡推理，C3=房间级布局。

**会产 0 题的情况**（不是 bug）：
- T14/T15/T16 在 0267_840790 场景跑——该场景没有 `front_normal`/`label_side` 标注。
- 任何模板在 "trigger 找不到 3 个满足条件的物体" 时都会产 0 题。

---

## 7. 系统里的四个注册表（不要混淆）

```
INSTANTIATORS         template_id → 工厂函数         （22 项，每模板 1 个）
INIT_VALIDATORS       YAML trigger 字段名 → 验证函数  （字段共享，模板复用）
CHOICE_REGISTRY       generator name → 选项生成函数   （生成器名共享）
PREDICATE_REGISTRY    predicate name → 几何判定函数   （24 项 = 12 函数 × snake/Pascal 双键）
```

简化项 Item 7 之后，**第一个表**（`INSTANTIATORS`）通过 `TemplateHandler` 与 `TemplateSpec` 打包：

```python
from spatial_training_room.task_generation.apl_tasks.template_handler import (
    get_template_handler, all_template_handlers,
)
h = get_template_handler("T05")
h.spec.template_id      # "T05"
h.instantiator(spec, ctx, rng)
```

另外三个表**故意**不打包到 `TemplateHandler` 里——它们是按"调用名"索引的共享查找表，多个模板复用同一个 predicate / choice / validator，强制 per-template 化只会引入重复。这是个**诚实重构**：能合的合，不能合的就明说不能合，并把理由写进代码注释。

### 7.1 全部 15 个 choice generator

源码：[task_generation/apl_tasks/choice_generators.py](../task_generation/apl_tasks/choice_generators.py)。YAML 写 `answer_choices_generator: <name>` 即调用。

| 名称 | 输出选项 |
|---|---|
| `binary_yes_no` | yes / no |
| `binary_same_different` | same / different |
| `binary_complete_incomplete` | complete / incomplete |
| `binary_continuous_separate` | continuous / separate |
| `pair_labels` | 题目里两个对象的 label 当二选 |
| `similar_aabb_labels` | 同尺寸/同类干扰项 + GT |
| `directional_choices` | left/right/front/back/up/down 子集 |
| `count_range` | "1" / "2-3" / "4-6" / "7+" 区间 |
| `count_range_or_zone_comparison` | 计数区间 + 房间分布 |
| `count_and_arrangement` | 计数 + 排布形态 |
| `contact_relations` | on / next_to / separate |
| `config_shapes` | triangle / line / cluster |
| `property_choices` | 模板自定义属性集 |
| `label_face_choices` | front / back / left / right（贴标面）|
| `back_face_property_choices` | T14 背面属性子集 |

**正确答案** 由 YAML 的 `gt_from` 字段决定（如 `gt_from: gt_answer` 直接读 instantiator 算好的；或 `gt_from: subject_label` 直接拿 instance 的某字段）。

---

## 8. 如何检查输出是不是合理的

### 8.1 看每条 JSONL 的关键字段
- `coverage == 1.0`：合格。`< 1.0`：最后一帧没满足所有 slot——通常是 beam search 步数不够，可考虑放大 `max_steps` 或检查 region 设计。
- `score`：和 `coverage` 不同，**包含 $\gamma^T$ 衰减**——`score=0.95`、`coverage=1.0`、`num_steps=1` 是理想；`score=0.4`、`coverage=1.0`、`num_steps=12` 说明走太远。
- `answer` 必须在 `choices` 里出现，否则数据写错了（实际线上从未触发，这是稳态约束）。
- `init_view ≠ target_view`：若相等说明 `_sample_failing_init` 失败回退（极少见，需排查 trigger 写得太苛刻）。

### 8.2 跑冒烟测试
```powershell
cd c:\Users\user\Desktop\code\spatial-training-room
C:\Users\user\miniconda3\python.exe testbench\smoke_test_template.py `
    --template T05 --scene C:\Users\user\Desktop\0267_840790 --n 5
```
看末尾打印的 `[smoke] produced N task(s).` 是否等于 `--n`。`SceneRequirementUnmet` 反复出现 → 该场景缺该模板所需的物体/标注，**不是 bug**（T14/T15/T16 在 0267 场景就是这种情况）。

### 8.3 跑系统级单测
```powershell
C:\Users\user\miniconda3\python.exe testbench\test_items_6_and_7.py
```
应该以 `ALL CHECKS PASSED` 结尾。这个文件覆盖：
- 22 个 `TemplateHandler` 都加载得到 `(spec, instantiator)`
- 合成 YAML 用 `checks:` 写法、用旧字段写法、两者混用三种情况都解析正确
- 4 类错误输入都抛带 template_id 的清晰 `ValueError`

### 8.4 看图（可视化）
```powershell
C:\Users\user\miniconda3\python.exe testbench\visualize_tasks.py `
    --scene C:\Users\user\Desktop\0267_840790 `
    --jsonl "out/batch/T05.jsonl" `
    --out out/vis
```
在 `out/vis/index.html` 里浏览：每张图含一张 top-down 地图（init 红、target 绿、轨迹蓝）加任务信息面板。眼睛能直接看到"初始位置确实拍不到证据"、"终点位置朝向参考物"。

### 8.5 单条任务 trace（教学/调试）
```powershell
C:\Users\user\miniconda3\python.exe testbench\trace_T05.py `
    --scene C:\Users\user\Desktop\0267_840790 `
    --jsonl out/batch/T05.jsonl --task-index 0 --out out/trace
```
产出 `summary.md`（每一步 $\Phi$ / slot 通过表）+ region 散点图 + 轨迹图 + $\Phi$ 曲线。**第一次理解专家搜索为什么这么走时强烈推荐跑一遍。**

---

## 9. 怎么用

### 9.1 安装
```powershell
cd c:\Users\user\Desktop\code\spatial-training-room
pip install -r requirements.txt
# 可选：要渲染真实 RGB 图
pip install torch gsplat plyfile imageio
```

### 9.2 单模板冒烟
```powershell
C:\Users\user\miniconda3\python.exe testbench\smoke_test_template.py `
    --template T05 --scene C:\Users\user\Desktop\0267_840790 --n 5
```

### 9.3 全量批跑
```powershell
C:\Users\user\miniconda3\python.exe testbench\sweep_all_templates.py `
    --scene C:\Users\user\Desktop\0267_840790 `
    --n 5 --out out/batch
```
每个 `T??` 产出一份 `out/batch/T??.jsonl`。22 个模板 × 5 道题在单场景约 15–25 分钟。

### 9.4 渲染真实 3DGS 图（init/target 各一帧）
```powershell
C:\Users\user\miniconda3\python.exe testbench\render_real_tasks.py `
    --scene C:\Users\user\Desktop\0267_840790 `
    --jsonl out/batch/T05.jsonl --out out/real --max 5
```
没有 CUDA 时 gsplat 会自动走 CPU 回退（很慢、质量低）。

### 9.5 写新模板的最短流程

1. 在 `task_generation/templates/p0/T34_my_thing.yaml` 写 YAML，**优先用 `checks:` 写法**。
2. 在 `task_generation/apl_tasks/template_active_generator.py` 加一个：
   ```python
   @register_instantiator("T34")     # 或等价的 @register_template("T34")
   def _instantiate_T34(spec, scene_ctx, rng):
       ...
       return {"target_id": ..., "target_label": ..., "gt_answer": ...}
   ```
3. 如果 trigger 用了新 YAML 字段，去 `init_validators.py` 注册：
   ```python
   @register_init_validator("my_new_field")
   def _v_my_new_field(value, spec, ti, cp, ct, hfov, ctx):
       return 1.0 if condition else float("-inf")
   ```
4. 如果选项规则新，去 `choice_generators.py` 注册：
   ```python
   @register_choice("my_choices")
   def _my_choices(ti, ctx, rng): return [...]
   ```
5. 跑 `smoke_test_template.py --template T34 --n 5` 验证。

---

## 10. 设计这套系统时被反复权衡的几件事

写在这里是为了下次有人想"为什么不直接 X"时有据可查：

1. **YAML 还是 Python 定义模板？** YAML——题目是给非程序员审稿用的，配置代码分离。
2. **predicate 为什么是 PascalCase？** 历史遗留；它们就是函数。简化项 Item 4 加了 snake_case 别名，但没有动 `def` 名字以零破坏旧调用。
3. **为什么有 `region_generator` 这一层而不直接全房间撒点？** 搜索可行性。整间房 10⁴ 候选 × N slot × N predicate 不可能跑完。
4. **为什么提交时判分而不路过判分？** 防作弊：扫一圈一定会路过合格视角。
5. **为什么 init validator 是按 YAML 字段名分发，不是按 template_id？** 多模板复用约束（`target_invisible_at_init` 至少有 5 个模板都要）。
6. **为什么 `TemplateHandler` 只合并 1/4 个 registry？** 其他 3 个是按调用名（predicate 名、generator 名、字段名）索引的共享表，**多个模板复用同一项**——按 template_id 打包会引入重复。诚实重构 > 强行统一。
7. **为什么 init validator 内部还是 5 档分值（1.0 / 0.5–0.7 / 0.1–0.3 / 0.0 / -inf）但对外只返回 `(bool, float)`？** Item 2 简化的是**对外契约**；内部多档评分是有信号的（"目标在侧后方"比"目标在头顶"更典型），保留它能让 init 选择更均衡，**只是不让外部代码看见**。

---

## 11. 走读路线（按这个顺序读源码）

如果你完全没看过这套代码，按这个顺序读：

1. [docs/GLOSSARY.md](GLOSSARY.md)——名词速查（一页）。
2. [task_generation/templates/p0/T05_actual_distance_comparison.yaml](../task_generation/templates/p0/T05_actual_distance_comparison.yaml)——读一份完整 YAML。
3. [evaluation/template_spec.py](../evaluation/template_spec.py)——YAML 怎么变 Python 对象，`checks:` 怎么投影。
4. [task_generation/apl_tasks/template_active_generator.py](../task_generation/apl_tasks/template_active_generator.py) 的 `_instantiate_T05`——一个完整 instantiator。
5. [evaluation/predicates.py](../evaluation/predicates.py)——12 个判定函数。
6. [task_generation/apl_tasks/init_validators.py](../task_generation/apl_tasks/init_validators.py) 的 `score_init_view`——三元组 API。
7. [evaluation/expert.py](../evaluation/expert.py)——beam search 主循环。
8. [task_generation/apl_tasks/template_handler.py](../task_generation/apl_tasks/template_handler.py)——简化版打包接口。
9. [testbench/trace_T05.py](../testbench/trace_T05.py)——把 1–7 全串起来打印每一步。

读完这 9 个文件，整个仓库等于已经看完。其他的都是边角料（QA 旧轨道、可视化、CLI 包装等）。

---

## 12. 人工标注工作流（仅 `front_normal` + `label_side`）

本项目当前默认**不要求人工标注 `completeness`**：T19 的 instantiator 直接给 `gt_answer="complete"`，因此人工标注重点是 `front_normal` 和 `label_side`。

### 12.1 两个字段分别是什么

- `front_normal`：物体"正面"在世界坐标中的单位向量，例如 `[0, 1, 0]`。常见判定优先级：屏幕面 > 主开口面 > 使用者正对面。
- `label_side`：文字/logo 所在面，使用离散枚举：`front/back/left/right/top/bottom`。

### 12.2 推荐标注流程（单场景）

1. 从 `labels.json` 导出待标物体清单（`id + label + AABB`）。
2. 给每个物体预生成 4~8 张参考图（top-down + 不同 yaw 的局部截图）。
3. 标注员逐个填写：
   - `front_normal`（可先用 8 方位按钮：N/NE/E/SE/S/SW/W/NW）；
   - `label_side`（6 选 1；无文字面可留空并打 `no_label_face=true`）。
4. 每 20 个物体做一次双人复核：有冲突的样本交由第三人裁定。
5. 回写 `labels.json` 后，立即跑：

```powershell
C:\Users\user\miniconda3\python.exe testbench\smoke_test_template.py `
    --template T14 --scene <scene_dir> --n 5
C:\Users\user\miniconda3\python.exe testbench\smoke_test_template.py `
    --template T15 --scene <scene_dir> --n 5
C:\Users\user\miniconda3\python.exe testbench\smoke_test_template.py `
    --template T16 --scene <scene_dir> --n 5
```

只要三者都能稳定产题，说明标注字段已经满足该场景的最小需求。
