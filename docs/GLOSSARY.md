# 项目术语表 / Glossary

本项目里反复出现的几个词、缩写、字段名，集中放在这里一次解释清楚。
阅读源码或 YAML 之前先扫一遍。

---

## 顶层概念

| 词 | 它真正的意思 |
|---|---|
| **APL** | Active Perception Loop（主动感知循环）。一种数据范式：模型先看一帧 → 决定动作 → 再看一帧 → 直到能回答问题。本仓库的题目都是为这套循环准备的。 |
| **template** | 一类题目的 YAML 定义（例如 `T05_actual_distance_comparison.yaml`）。每个 template 描述：题面文字模板、答案怎么算、需要看到什么。 |
| **template_id** | 模板短码（`T01`–`T33`）。文件名前缀就是它。短码本身没含义，要看文件名后半段。 |
| **subclass** | 形如 `C1.2 / C2.3` 的子类码。和 `priority` 一样属于纯标签，对生成逻辑不影响。 |
| **priority** | `P0` / `P1` / `P2`。只表示这个模板是不是"必须做"的核心题，跟难度无关。 |
| **task_instance** | 模板里那些占位符（`{target_id}`、`{room_a_id}` 等）被填上具体 ID 后的字典。一个 task_instance = 一道具体题目的"骨架"。 |
| **instantiator** | 给某个模板挑出具体 ID 的工厂函数，叫 `_instantiate_T05` 这样的名字。读起来等同于 "build_task_instance"。 |

## "条件" 类的几个并行词汇

历史原因，源码里"某一帧画面要满足什么条件"被叫了三个不同的名字：

| 出现位置 | 名字 | 实际作用 |
|---|---|---|
| YAML 顶层 `scene_requirements:` | scene_requirement | 在挑 `task_instance` 之前先查（"这场景里是不是至少有两个房间"）|
| YAML 顶层 `trigger:` | trigger field | 在挑 init view 时查（"主体目标在起点不能看见"）。每个字段由 `init_validators.py` 解释。 |
| YAML `evidence_slots:` 里 | predicate | 在挑 target view 时查（"主体目标当前是可见的"）|

简短记忆：**scene → trigger → predicate**，对应"选场景 → 选起点 → 选终点"三个阶段。

### 统一写法：`checks:`（推荐）

新 YAML 可以用单一的 `checks:` 列表代替上面三个字段，加载时由 `evaluation/template_spec.py::_project_checks` 自动展开到旧字段——**旧 YAML 不用改**，两种写法可以混用。示例：

```yaml
checks:
  - {when: scene,  name: min_objects, args: 3}
  - {when: init,   name: target_invisible_at_init, args: true}
  - {when: target, slot: AB_view,
     name: PairVisible, args: {obj_a: "{{subject_id}}", obj_b: "{{ref_b_id}}"}}
```

- `when: scene` → 合并进 `scene_requirements`
- `when: init` → 合并进 `trigger`
- `when: target` → 追加到指定 `slot` 的 `predicates`（不写 `slot:` 则进第一个 slot）
- 显式 `scene_requirements` / `trigger` / `evidence_slots` 字段优先（用 `setdefault` 合并）。

## init view / target view / expert trajectory

| 词 | 含义 |
|---|---|
| **target view** | "能直接回答问题"的视角。每个 evidence slot 描述的就是这样一帧。 |
| **init view** | 起点视角。**故意挑一个回答不了问题的视角**，这样才需要 active perception。 |
| **expert trajectory** | 从 init 走到 target 的标准答案路径，由 `evaluation/expert.py` 的 beam-search 找出。 |
| **evidence slot** | "看到 X 的一帧画面" 的描述：region 生成器 + 一组 predicate。同一个题目可以有多个 slot（例如 T05 要分别看到 A+B 和 A+C 两对）。 |

## predicate 是函数，不是类

`evaluation/predicates.py` 里 PascalCase 的名字（`Visible`、`PairVisible`、`BearingWithin`…）历史上是函数。现在已经提供 snake_case 版本（`is_visible`、`pair_visible`、`bearing_within`）作为推荐用法。YAML 中两种写法都仍然可用。

## init validator 的返回约定

`task_generation/apl_tasks/init_validators.py` 里每个验证函数的签名是：

```python
def validator(value, spec, task_instance, cp, ct, hfov, scene_ctx) -> (bool, float):
    # bool  = 是否通过硬性条件（False 即拒绝这个候选）
    # float = 偏好分（≥0，越大越优先）
```

历史代码里见到的 `-inf` / `0.5` / `1.0` 五档约定已经废弃，只剩 "通过 / 不通过 + 偏好分" 两个维度。

## 各种注册表

源码里有几个名字相近的注册表，作用要分清：

| 注册表 | 在哪儿 | 用途 |
|---|---|---|
| `INSTANTIATORS` | `apl_tasks/template_active_generator.py` | template_id → 工厂函数 |
| `INIT_VALIDATORS` | `apl_tasks/init_validators.py` | YAML trigger 字段名 → 验证函数 |
| `CHOICE_REGISTRY` | `apl_tasks/choice_generators.py` | YAML `answer_choices_generator:` → 选项生成函数 |
| `PREDICATE_REGISTRY` | `evaluation/predicates.py` | YAML predicate name → 函数 |

### TemplateHandler：按 template_id 拿东西的快捷口

上面 4 个表里，**只有 `INSTANTIATORS` 是按 `template_id` 索引**的；其他 3 个是按“调用名”索引的共享查找表（多个模板复用同一个 predicate / choice / validator），强行按 template_id 打包会重复。

为了避免每次调用者写两次查找（`load_template(tid)` + `INSTANTIATORS[tid]`），`task_generation/apl_tasks/template_handler.py` 提供一个轻量包装：

```python
from spatial_training_room.task_generation.apl_tasks.template_handler import (
    TemplateHandler, get_template_handler, all_template_handlers, register_template,
)

h = get_template_handler("T05")        # 一次拿到 spec 和 instantiator
h.template_id, h.spec, h.instantiator
```

新模板可以用 `@register_template("T35")`，它是 `@register_instantiator("T35")` 的别名；旧装饰器仍然可用。

## 缩写 / 易误读符号

| 符号 | 解释 |
|---|---|
| `cp` | camera position (世界坐标 3D 点) |
| `ct` | camera target (世界坐标 3D 点，相机看向的点) |
| `hfov` | horizontal field of view（度） |
| `bmin / bmax` | AABB 的两个角点 |
| `Cov`, `γ^T` | 在 `Score = correct · min(1, Cov/min_cov) · γ^T` 公式里：覆盖率与每步衰减 |
| `SceneRequirementUnmet` | 异常类型。表示"场景本身不满足这个 template 的硬性需求"，**不是 bug**，是数据问题。 |
