# Spatial Training Room

空间推理任务数据工厂，支持 **双轨** 数据生成：静态 QA 问答（`qa`）与动作-感知循环（APL, Action-Perception Loop）序列任务（`apl_passive` / `apl_active`）。

---

## 项目结构

```
spatial-training-room/
├── core/                          # 核心数据类型 & 场景上下文
│   ├── data_types.py              # ViewState, QATaskItem, APLPassiveTaskItem, APLActiveTaskItem
│   ├── scene_context.py           # SceneContext：加载场景几何、遮挡查询
│   └── task_base.py               # BaseTaskGenerator 抽象基类
├── action_space/                  # 动作空间定义 & 执行器
│   ├── action_primitives.py       # ActionPrimitive 枚举 & ActionConfig
│   ├── action_executor.py         # ViewStateExecutor（纯 numpy）
│   └── action_sequences.py        # GoalDirectedPlanner, SequenceValidator
├── task_generation/               # 任务生成器
│   ├── base_generator.py          # BaseAPLGenerator（共用脚手架）
│   ├── qa_tasks/
│   │   └── qa_generator.py        # QAGenerator（包装 bench_generation）
│   └── apl_tasks/
│       ├── apl_types.py           # 枚举 & NL 模板
│       ├── passive_generator.py   # APLPassiveGenerator（指令跟随）
│       └── active_generator.py    # APLActiveGenerator（问题驱动导航）
├── bench_generation/              # 原始 QA 批量生成（底层实现）
├── configs/
│   ├── apl_config.yaml            # APL 生成参数
│   └── qa_config.yaml             # QA 生成参数
├── sampler/                       # 视角采样工具
├── motion/                        # ViewManipulator（c2w 矩阵操作）
├── utils/
│   └── occlusion.py               # AABB, 射线投射, 遮挡计算
└── run_factory.py                 # 统一 CLI 入口
```

---

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt   # numpy, shapely, pyyaml 等
```

### 生成数据

```bash
# APL 双轨（passive + active），单个场景
python run_factory.py \
    --scenes /path/to/ViewSuite/data/0013_840910 \
    --mode apl \
    --config configs/apl_config.yaml \
    --out-dir ./out \
    --max-items 100

# QA 任务
python run_factory.py \
    --scenes /path/to/ViewSuite/data/0013_840910 \
    --mode qa \
    --config configs/qa_config.yaml \
    --out-dir ./out

# 全部（qa + apl_passive + apl_active），批量场景目录
python run_factory.py \
    --scenes /path/to/ViewSuite/data \
    --mode all \
    --out-dir ./out \
    --max-items 200
```

生成的 JSONL 文件保存在 `--out-dir`，格式为：

```
{scene_name}_qa.jsonl
{scene_name}_apl_passive.jsonl
{scene_name}_apl_active.jsonl
```

---

## 任务类型

### QA（静态问答）

| 类型 | 说明 |
|------|------|
| `object_count_mca` | 可见物体计数 |
| `nearest_object_mca` | 最近物体判断 |
| `object_object_distance_mca` | 物体间距离 |
| `relative_position_mca` | 相对方位 |
| `next_frame_mca` | 下一帧预测 |

### APL Passive（指令跟随导航）

模型需要执行动作序列以满足给定指令。

| 类型 | 说明 |
|------|------|
| `distance_absolute` | 移动到距目标物体指定距离 |
| `direction_face` | 转向正对目标物体 |
| `relative_position` | 移动到目标物体的左/右/前/后 |

### APL Active（问题驱动导航）

模型需要主动导航以回答空间问题。

| 类型 | 说明 |
|------|------|
| `visibility_single` | "你的左/右/后方是什么？"（需要转向） |
| `visibility_hidden` | "anchor 的 [方位] 是什么？"（需要移动到可见位置） |
| `next_action` | "为了看到 X，下一步应该怎么做？" |
| `spatial_distance` | "A 离 B 还是 C 更近？" |

---

## 动作空间

离散动作集合（`ActionPrimitive`）：

| 动作 | 键值 | 默认步长 |
|------|------|---------|
| `move_forward` | `w` | 0.5 m |
| `move_backward` | `s` | 0.5 m |
| `move_left` | `a` | 0.5 m |
| `move_right` | `d` | 0.5 m |
| `turn_left` | `q` | 45° |
| `turn_right` | `e` | 45° |
| `look_up` | `r` | 15° |
| `look_down` | `f` | 15° |

可在 `configs/apl_config.yaml` 中调整步长和序列长度上限。

---

## 遗留接口（bench_generation）

原始批量 QA 生成器仍可独立使用：

```bash
python -m bench_generation.qa_batch_generator \
  --scene /path/to/ViewSuite/data \
  --out ./tmp/qa_batch_test.jsonl \
  --out-dir ./try \
  --max_items 10 \
  --max_items_per_view 2 \
  --render

# 指定问题类型
python -m bench_generation.qa_batch_generator \
  --scene /path/to/ViewSuite/data \
  --out ./tmp/qa.jsonl \
  --out-dir ./try \
  --max_items 10 \
  --max_items_per_view 2 \
  --question_type action_next_frame_mca
```

视角采样工具（`sampler`）：

```bash
# 对所有场景的所有物体生成图片和 meta.json
python -m sampler.batch_generate_views \
  --scenes_root /data \
  --out ./out_views \
  --per_room_points 12
```

可调参数：`occ_ratio`（遮挡阈值）、`max_height` / `min_height`、`per_room_points`、`min_dist` / `max_dist`。
