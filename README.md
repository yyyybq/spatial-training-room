# Spatial Training Room

空间推理任务数据工厂。把一个 3D 室内重建（Gaussian Splatting 场景）+ 物体 AABB 标注 + 房间多边形作为输入，自动批量产出**主动感知 (Active Perception Learning, APL)** 训练样本：每条样本是一道带 `(初始视角, 专家动作序列, 最终视角, 问题, 答案)` 的 4D 题。

---

## 1. 项目目标：训练能"主动看"的 3D 室内空间智能体

普通 VQA 给一张图问问题。本项目要求的是 **主动感知**：

- 智能体被放在一个真实重建的 3D 房子里（3DGS 场景）；
- 初始视角故意被设成**问不出答案**（目标太远、被遮挡、不在视野内）；
- 智能体必须自己**走/转/俯仰**到合适的视角，看到足够证据后才能作答；
- 训练数据是一整条 `(init_view, actions, target_view, question, answer)` 轨迹，不是单张图。

整套 `spatial-training-room/` 就是**自动批量生产这种带轨迹题目的数据工厂**。

---

## 2. 三层架构

```
templates/p0/*.yaml          ← 题目"骨架"，描述本类题需要什么证据
   │
   ▼
apl_tasks/template_active_generator.py  ← Python "实例化器"，把骨架填上具体对象/房间
   │
   ▼
evaluation/                  ← 通用引擎：predicate, region 采样, 专家搜索, 打分
   │
   ▼
out/batch/T??.jsonl          ← 每行一道带专家轨迹的题
```

`testbench/sweep_all_templates.py` 把以上串起来跑批量。

---

## 3. 关键新概念

### 3.1 `SceneContext`

加载一个场景目录后（例 `0267_840790/`）生成的内存对象（[core/scene_context.py](core/scene_context.py) + [core/scene_context_ext.py](core/scene_context_ext.py)）：

- **物体列表**：每个物体有 `id, label, aabb(中心+尺寸), room_id`
- **房间多边形** + 房间之间的 **portal**（门洞，带宽度、位置）
- **遮挡查询**：给一个相机位姿，返回"对象 X 是否被挡住、占多少像素"
- **投影函数**：给位姿 + FOV，返回"对象 X 的 8 个 AABB 角有几个落在视野里"

所有 predicate / 区域采样都在它上面跑。

### 3.2 模板 (Template) — 一个 YAML 文件

一个 YAML 描述"这一类题怎么生成、需要看到什么、怎么判分"。以 [task_generation/templates/p0/T05_actual_distance_comparison.yaml](task_generation/templates/p0/T05_actual_distance_comparison.yaml) 为例：

```yaml
template_id: T05
question_templates:
  - "Is the {subject_label} closer to the {ref_b_label} or to the {ref_c_label}?"
answer_type: categorical
answer_choices_generator: pair_from_triple
gt_from: aabb_center_distance_comparison

trigger:
  min_dist_ratio: 1.3
  max_both_pairs_visible_at_init: false

evidence_slots:
  - slot_id: AB_view
    region_generator: around_pair
    region_args: {obj_a: "{{subject_id}}", obj_b: "{{ref_b_id}}", dist_min: 0.8, dist_max: 4.0}
    predicates:
      - {name: PairVisible, args: {obj_a: "{{subject_id}}", obj_b: "{{ref_b_id}}", min_corners: 4, max_occ: 0.30}}
      - {name: Centered,    args: {obj: "{{subject_id}}", theta_deg: 40}}
  - slot_id: AC_view
    region_generator: around_pair
    region_args: {obj_a: "{{subject_id}}", obj_b: "{{ref_c_id}}", ...}
    predicates: [...]

coverage_aggregator: mean
min_coverage_for_credit: 1.0

action_config: {move_m: 0.5, turn_deg: 45, look_deg: 15}
max_steps: 12
gamma: 0.95
```

#### `trigger` — "这道题在该场景里值不值得出"的门槛

一组**可生成性条件**。生成器选好候选变量后，先用 trigger 检查：

- T05 的 `min_dist_ratio: 1.3`：A→B 与 A→C 距离必须相差 ≥30%，否则没有"明显更近"的答案，这道题就退化成猜硬币，不出。
- `max_both_pairs_visible_at_init: false`：要求**初始视角不能同时看到两对**，否则一眼就能回答，没有训练价值。

trigger 不通过 → 跳过这个候选，重抽。反复重抽都不通过 → 这个场景出不了这个模板的题。

#### `evidence_slots` — 必须采集的"证据片段"

一道题被判合格，智能体**最终视角**必须**同时**满足所有 slot 的 predicates。一个 slot ≈ "必须拍到的一帧"应满足的条件集合。

T05 有两个 slot，因为 A、B、C 三个物体一般不可能同时入镜；要正确回答必须分别去看 (A,B) 和 (A,C)。

slot 内字段：

- **`region_generator`**：定义"哪片站位 + 朝向区域"在几何上有可能满足这个 slot。例如 `around_pair` 在两对象中点周围采位姿、朝向中点。这是给搜索算法**剪枝**用的——不要在整个房子盲搜。
- **`predicates`**：判定一个视角是否真的"看到了"。常用的 12 个在 [evaluation/predicates.py](evaluation/predicates.py)：
  - `PairVisible(a, b, min_corners=4, max_occ=0.30)`：两对象 AABB 各至少 4 个角投影在视野内，遮挡比例 ≤30%。
  - `Centered(obj, theta_deg=40)`：对象中心方向角与相机朝向夹角 ≤40°。
  - `Visible / InRoom / PortalVisible / OcclusionPairVisible / ...`
- **`threshold: 1.0`**：slot 内 predicate 满足比例下限，1.0 = 全部 predicate 都得过。

#### `coverage_aggregator` / `min_coverage_for_credit`

所有 slot 各自算出 0/1 后按 `mean` 或 `all`(乘积) 聚合得到 `coverage ∈ [0,1]`，低于 `min_coverage_for_credit` → 得分清零。

#### `action_config` / `max_steps` / `gamma`

智能体可用的离散动作集（前进 0.5 m、左右转 45°、俯仰 15°，外加 SUBMIT），最多 `max_steps` 步，折扣因子 `gamma`。

最终打分（[evaluation/scorer.py](evaluation/scorer.py)）：

$$
\text{Score} = \mathbb{1}[\hat a = a^*] \cdot \min\!\left(1, \frac{\text{Coverage}}{\text{min\_coverage}}\right) \cdot \gamma^{T}
$$

即"答对 × 证据足够 × 步数惩罚"。$\gamma^T$ 鼓励用尽量短的轨迹完成。

#### `scene_requirements`

预过滤：例如要求场景里至少 3 个物体、至少 2 个房间、portal 数 ≥1，否则连尝试都不尝试。

### 3.3 Instantiator — 把 YAML 骨架填上具体对象

YAML 里写的是 `{subject_id}`, `{ref_b_id}` 这类占位符；选谁由 Python 函数决定。[task_generation/apl_tasks/template_active_generator.py](task_generation/apl_tasks/template_active_generator.py) 里每个模板对应一个 `_instantiate_TXX(spec, scene_ctx, rng)`：

1. 从 `scene_ctx` 抽候选对象/房间组合；
2. 用 trigger 检查（距离比、label 不同、同房间等）；
3. 通过 → 返回 `{subject_id: 17, ref_b_id: 4, ref_c_id: 9, subject_label: "sofa", ...}`，骨架里的占位符会被替换。

不通过 → 重抽，最多重抽 N 次都失败就抛 `SceneRequirementUnmet`。

### 3.4 Region Generator ([evaluation/region_generators.py](evaluation/region_generators.py))

给定 slot 参数，在 3D 房间里**采样一批候选视角 `(x, y, z, yaw, pitch)`**。例：

- `around_pair(obj_a, obj_b, dist_min, dist_max)`：在两物体中点周围一个环形带里采位置，朝中点；
- `behind_occluder(fg, bg)`：在能看到 fg 挡住 bg 的位置采；
- `inside_room(room_id)`：房间内随机；
- 共 16 个。

作用：把"可行解空间"从整间屋子缩到几十个候选位姿，让后面的搜索可行。

### 3.5 `init_view` 与 `target_view`

- **`target_view`**：用 region_generator 采 + predicate 筛选出的、**满足所有 slot** 的视角。证明这道题**有解**。
- **`init_view`**：另选一个**故意让 predicate 不通过**的视角作为起点，由 `_sample_failing_init` 保证（看不到目标对、被挡、不在房间里等）。这就是"问题在初始时回答不了"的强制条件。

没有这一步，题目会退化成"睁眼就能答"，训练不出主动感知能力。

### 3.6 专家轨迹 (Expert Trajectory) 与搜索

有了 `init_view` 和"什么视角算合格 (slot predicates)"，就要算**最少几步能从 init 走到合格 view**。这是 [evaluation/expert.py](evaluation/expert.py) 做的 beam search：

- 状态 = `ViewState(x, y, z, yaw, pitch)`；
- 动作 = `ActionPrimitive` 离散集合（前后左右、左右转、俯仰、submit）；
- 邻居 = 当前状态依次套用每个动作，加碰撞检查；
- 节点剪枝键：`(round(x/0.25), round(y/0.25), round(yaw/22.5°))` —— 同格同朝向只留一个，避免爆炸；
- 启发 / 评分：用 **potential function** $\Phi(s)$（在 [evaluation/potential.py](evaluation/potential.py)），它是"当前视角离满足 slot predicate 的距离的负值"，越大越好；
- 每步塑形奖励：$r_t = \Phi(s_{t+1}) - \Phi(s_t)$，最终步加上 episode score；
- Beam 按累计回报排序保留 top-K，直到某条路径到达"提交时所有 slot 满足"的状态。

结果：一个 `expert_trajectory: List[ViewState]` —— 这道题的**金标准动作序列**。强化学习/模仿学习训练时它既是行为克隆目标，也是 reward 计算参照。

> **Submit-time quality**（提交时判分）：判 coverage 时**只看 trajectory 最后一帧**，路过合格视角不算。防止智能体扫一圈作弊。

### 3.7 Choice Generator ([task_generation/apl_tasks/choice_generators.py](task_generation/apl_tasks/choice_generators.py))

YAML 的 `answer_choices_generator: pair_from_triple` 指向注册好的 14 个生成器之一。例：

- `binary_yes_no` → `["yes","no"]`
- `pair_from_triple` → 用 ref_b_label 和 ref_c_label 做二选一
- `count_range` → GT ± 1, ±2 做四选一
- `similar_aabb_labels` → 挑外形/类别接近的 label 做干扰项

把开放生成转成多项选择，便于自动判分。

### 3.8 一条 task 的最终样子（JSONL 一行）

```jsonc
{
  "task_id": "T05_0267_840790_007",
  "template_id": "T05",
  "subclass": "C1.2",
  "question": "Is the sofa closer to the lamp or to the bookshelf?",
  "choices": ["lamp", "bookshelf"],
  "answer": "lamp",
  "init_view":   {"position": [...], "target": [...], "forward": [...]},
  "target_view": {"position": [...], "target": [...], "forward": [...]},
  "expert_trajectory": [ViewState, ...],
  "action_descriptions": ["Turn right 45°", "Move forward 0.5m", ..., "Stop"],
  "quality_spec": {"evidence_slots": [...], "gamma": 0.95, "min_coverage_for_credit": 1.0},
  "coverage": 1.0,
  "score": 0.86,
  "num_steps": 4
}
```

---

## 4. 一次生成全流程（7 步）

`generate_for_template(template_id, n)`：

1. **加载模板**：`load_template("T05")` → `TemplateSpec`。
2. **实例化**：调 `_instantiate_T05(spec, scene_ctx, rng)`，挑出 subject/ref_b/ref_c 三个物体；trigger 检查（距离比 ≥1.3、label 互异、同房间）。
3. **可行性证明**：对每个 evidence_slot：
   - 用 `region_generator` 采 N 个候选位姿；
   - 用 `slot_satisfied_at(view, slot, scene_ctx)` 检查 predicates；
   - 至少存在一个候选满足 → 该 slot 可达。所有 slot 都可达 → 该实例可行。
4. **选 init_view**：`_sample_failing_init` 随机采视角并要求**至少一个 slot 不满足**（保证一开始问不出）。
5. **跑专家搜索**：`find_expert_trajectory(init_view, slots, scene_ctx, max_steps, gamma)` → `expert_trajectory` 和 final view。
6. **打分**：`compute_coverage` 在 final view 上算 coverage；`episode_score` 算 score。
7. **打包**：把 `{subject_label}` 等占位符填入问题模板 → `question`；调 `build_choices` → `choices`；按 `gt_from` 计算 `answer`；落盘 JSONL 一行。

`sweep_all_templates.py` 把第 1–7 步对 22 个模板 × n 次重复跑一遍。

---

## 5. 22 个模板分别测什么空间能力

> 通用：所有模板都遵循"初始看不见 → 走到位看见 → 提交"的主动感知范式。下面只讲**这道题独有的认知考点**。

| 模板 | 名字 | 考的空间能力 | trigger 关键 | slot 关键 predicates |
|---|---|---|---|---|
| T01 | category_recognition | 远处看不清的物体走近识别**类别** | 至少 1 个对象 init_view 看不清 | Centered + Visible 近距 |
| T04 | actual_size_comparison | 排除近大远小后比较**真实体积** | 体积比≥阈值 + 初始视角欺骗 | 两个 slot 拍 A、B 近距特写 |
| T05 | actual_distance_comparison | 同参考物到两目标的**真实距离**谁更近 | 距离比≥1.3 | PairVisible(A,B) 与 PairVisible(A,C) |
| T06 | clearance_assessment | 物体能否穿过最窄通道 | portal 宽度 vs 对象宽度 | 看到 portal + 看到对象 |
| T08 | configuration_judgment | 三对象**几何构型**（三角/共线）| 三对象同房间、label 互异 | 三个对象各自清晰 |
| T11 | single_multiple_split | 远看似一个其实是 **1 还是 2** | 存在视角依赖的歧义对 | 近距+侧向 slot 区分粘连 |
| T13 | post_occlusion_continuation | 被挡延伸是**同物还是另物** | 存在 fg-bg 遮挡对 | OcclusionPairVisible + 绕到背后 |
| T14 | back_face_acquisition | 看物体**背面**（需 `front_normal` 标注）| — | — |
| T15 | label face/side | 标签朝向哪面（需 `label_side` 标注）| — | — |
| T16 | front_back_difference | 正反面差异（需相应标注）| — | — |
| T17 | post_occlusion_existence | 被挡的位置**后面有没有东西** | 是否存在 bg | 走到 fg 侧后能看到 bg 区域 |
| T18 | post_occlusion_category | 被挡的是什么类别 | bg label 与干扰类区分得开 | Centered(bg) |
| T19 | post_occlusion_completeness | 被挡的对象**是否完好** | 需 completeness 标注 | 走到 bg 侧 |
| T20 | local_bearing_search | 周围 360° **方位搜索**到某物 | subject 必须在 init 附近可达 | Centered(subject) |
| T21 | local_nearest_target | A、B 哪个**离参考物更近** | 同房间、label 互异、距离差显著 | PairVisible(A, ref), PairVisible(B, ref) |
| T23 | cross_room_existence | 另一房间里**有没有 X** | 选定 target_room + label | InRoom + Visible |
| T24 | portal_direction | 通往房间 X 的**门在哪个方向** | portal 可达 | PortalVisible + Centered |
| T26 | occluded_counting | 含被遮挡在内**数总数** | 同类对象 ≥2 且 init 漏看 | 多视角覆盖全部实例 |
| T27 | zone_counting | 某/多个房间内 X **共几个** | zone 内确有 X | 进入各房间分别拍到 |
| T29 | contact_relationship | A 放在 B **上面/旁边/分离** | AABB 顶/侧关系明确 | 侧面 slot 看清接触点 |
| T32 | connectivity_judgment | 两房间**直接/经某房间**连通 | portal 图存在路径 | 沿走廊取证 |
| T33 | passage_passability | 通道**够不够某尺寸通过** | portal 宽度 vs 阈值 | PortalVisible + 对象近景 |

T14/T15/T16 在场景 `0267_840790` 里产不出题，**不是 bug，是该场景缺对应几何/语义标注**（front_normal、label_side 等）。换标注更全的场景就能出。

---

## 6. 数据为什么这样设计：训练目标决定

这套数据训练的模型最终要学会：

1. **看到题目就规划"我要去看什么"**（隐式学到 evidence_slots 的概念）；
2. **执行最短动作序列**（$\gamma^T$ 惩罚长轨迹）；
3. **只在证据充足时回答**（提交时判分，路过不算）；
4. **区分真实几何与单帧视觉错觉**（T04/T05/T11 这种"近大远小""粘连分离"题）。

所以 YAML 里出现的 `trigger / evidence_slots / region_generator / predicate / coverage / gamma` 不是凭空造的概念，每一个都对应训练目标里的一条约束：

| 概念 | 对应训练目标 |
|---|---|
| `trigger` | 不出弱智题 |
| `evidence_slots` | 定义什么叫"看够了" |
| `region_generator` | 把搜索空间缩到可解 |
| `predicate` | 几何上判定单帧是否合格 |
| `expert trajectory + γ^T` | 教模型走最短路 |
| `init_view` 强制失败 | 强迫模型动起来 |

---

## 7. 项目结构

```
spatial-training-room/
├── core/                          # 核心数据类型 & 场景上下文
│   ├── data_types.py              # ViewState, APLActiveTaskItem, ...
│   ├── scene_context.py           # SceneContext：加载场景几何、遮挡查询
│   └── scene_context_ext.py       # 房间/portal/投影 扩展
├── action_space/                  # 动作空间定义 & 执行器
│   ├── action_primitives.py       # ActionPrimitive 枚举 & ActionConfig
│   └── action_executor.py         # ViewStateExecutor（纯 numpy）
├── evaluation/                    # 模板引擎
│   ├── template_spec.py           # YAML 解析
│   ├── predicates.py              # 12 个 predicate
│   ├── region_generators.py       # 16 个 region 采样器
│   ├── coverage.py                # coverage 聚合
│   ├── potential.py               # Φ(s) 势函数
│   ├── expert.py                  # beam-search 专家
│   ├── scorer.py                  # episode_score
│   └── quality_overrides.py       # tier-2 callable 覆盖
├── task_generation/
│   ├── apl_tasks/
│   │   ├── template_active_generator.py  # 19 个 _instantiate_Txx
│   │   └── choice_generators.py          # 14 个选项生成器
│   └── templates/
│       ├── p0/*.yaml              # 19 个 P0 模板
│       └── p1/*.yaml              # 3 个 P1 模板
├── testbench/
│   ├── sweep_all_templates.py     # 批量生成所有模板
│   ├── visualize_tasks.py         # 2D top-down 任务可视化
│   ├── render_real_tasks.py       # 用 3DGS 渲染真实图像（gsplat）
│   └── trace_T05.py               # T05 单模板端到端 trace（见 §9）
├── bench_generation/              # 原始 QA 批量生成（旧）
├── configs/                       # 默认参数 YAML
└── run_factory.py                 # 统一 CLI 入口
```

---

## 8. 批量生成与可视化

```bash
# 安装依赖
pip install -r requirements.txt   # numpy, shapely, pyyaml, matplotlib
# 可选：渲染真实图像
pip install torch gsplat plyfile imageio

# 批量产 22 模板 × 5 道题
python testbench/sweep_all_templates.py \
    --scene C:/Users/user/Desktop/0267_840790 \
    --n 5 \
    --out out/batch

# 2D top-down 可视化（生成 PNG + HTML gallery）
python testbench/visualize_tasks.py \
    --scene C:/Users/user/Desktop/0267_840790 \
    --jsonl "out/batch/*.jsonl" \
    --out out/vis

# 真实 3DGS 渲染（init / target 视角图）
python testbench/render_real_tasks.py \
    --scene C:/Users/user/Desktop/0267_840790 \
    --jsonl out/batch/T05.jsonl \
    --out out/real \
    --max 5
```

输出：

- `out/batch/T??.jsonl` — 每行一道任务
- `out/vis/index.html` — 缩略图组成的 gallery（每张含 top-down 地图 + 任务信息面板）
- `out/real/T??/T??_task_XXX_{init,target}.png` — 渲染的 RGB 图

---

## 9. T05 端到端 Trace（教学用）

要直观看到 trigger → region 采样 → init/target → expert 每一步的 $\Phi$ 变化：

```bash
python testbench/trace_T05.py \
    --scene C:/Users/user/Desktop/0267_840790 \
    --jsonl out/batch/T05.jsonl \
    --task-index 0 \
    --out out/trace
```

会在 `out/trace/T05_task_000/` 下产出：

- `summary.md` —— trigger 检查表 + 步骤逐行 (`t | action | Φ(s) | ΔΦ | slot1 | slot2`)
- `01_regions.png` —— 两个 slot 的 region_generator 采样点（绿=通过 predicate，红=未通过）
- `02_trajectory.png` —— top-down 上画 init→...→target 每一步位姿与 FOV
- `03_phi_curve.png` —— $\Phi(s_t)$ 随 $t$ 的折线图

---

## 10. CLI 入口（旧）

`run_factory.py` 是较早的混合入口，覆盖 QA + APL passive + APL active 三种轨道：

```bash
python run_factory.py \
    --scenes /path/to/scene_dir \
    --mode template \
    --template-id T05 \
    --out-dir ./out \
    --max-items 50
```

---

## 11. 已知限制

- **CUDA 必要性**：gsplat 原生光栅化需要 CUDA，否则 `render_real_tasks.py` 走 CPU 回退（质量低）。
- **场景标注覆盖度**：T14/T15/T16/T19 需要 `front_normal / label_side / completeness` 等额外标注，0267 场景不具备 → 产 0 题。
- **T06 慢**：clearance 计算每个候选 portal 都要做几何裁剪，单题可能 >10 分钟，建议先用 `--templates` 过滤掉。

设计审计表见 [TEMPLATE_ACTIVE_README.md](TEMPLATE_ACTIVE_README.md)。
