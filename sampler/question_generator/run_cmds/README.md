每个脚本模板位于本目录下，文件名 `<module>.sh`。

使用方法：
  - 编辑每个 `<module>.sh`，将模板命令替换为所需的具体命令行参数。
  - 在需要的地方使用 `$COMMON_ARGS` 来引用 `run_all_generators.sh --run --common '...'` 中传入的共用参数。

示例：
  #!/usr/bin/env bash
  set -euo pipefail
  COMMON_ARGS=${COMMON_ARGS:-""}
  # Example from source: python -m Data_generation.sampler.question_generator.view_rotation_mca --scenes_root ...
  python -m Data_generation.sampler.question_generator.view_rotation_mca $COMMON_ARGS --out /tmp/view.json

不要直接删除模板文件；脚本 `run_all_generators.sh --generate` 会在文件不存在时生成模板，但不会覆盖已存在文件。


# 在 run_all_generators.sh 所在目录运行
./run_all_generators.sh --run --common '--scenes_root /data/liubinglin/jijiatong/ViewSuite/data/test --out-dir /data/liubinglin/jijiatong/ViewSuite/Data_generation/0267 --per_room_points 12'