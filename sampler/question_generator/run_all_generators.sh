#!/usr/bin/env bash
set -euo pipefail

# 批量生成并运行 Data_generation/sampler/question_generator 下的所有生成脚本命令模板
#
# 用法:
#   1. 先生成命令模板（会写入 run_cmds/ 下的 per-file scripts）
#      ./run_all_generators.sh --generate
#   2. 编辑每个模板文件 run_cmds/<module>.sh，修改/添加具体参数。
#      在模板中可以使用环境变量 $COMMON_ARGS 来指定所有脚本共用的参数。
#   3. 运行所有命令：
#      ./run_all_generators.sh --run --common '--scenes_root /path/to/data --out-dir /tmp/out' 
#
# 脚本行为：
#  - 会在当前目录下创建目录 run_cmds/，每个 <module>.py 生成一个 run_cmds/<module>.sh 模板（若已存在则不覆盖）
#  - 模板包含可编辑的命令行；脚本会按字母顺序运行 run_cmds/*.sh
#  - 若模板中存在首行注释中包含示例命令（来自 .py 顶部 docstring），会把该示例写入模板注释供参考

HERE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$HERE_DIR"
CMD_DIR="$SRC_DIR/run_cmds"

function usage() {
  cat <<EOF
Usage: $0 [--generate] [--run] [--common '<common args>']

Options:
  --generate    : generate per-file templates under run_cmds/ (won't overwrite existing files)
  --run         : run all templates found under run_cmds/ (must be executable)
  --common STR  : common args string to export as COMMON_ARGS for all scripts when running
  --help        : show this help

Typical workflow:
  $0 --generate
  # edit run_cmds/*.sh, insert per-file args and use \$COMMON_ARGS for shared options
  $0 --run --common "--scenes_root /path/to/data --out-dir /tmp/out"
EOF
}

GEN=0
RUN=0
COMMON_ARGS=""
FORCE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --generate) GEN=1; shift ;;
    --run) RUN=1; shift ;;
    --common) COMMON_ARGS="$2"; shift 2 ;;
    --force) FORCE=1; shift ;;
    --help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 1 ;;
  esac
done

mkdir -p "$CMD_DIR"

if [[ $GEN -eq 1 ]]; then
  echo "Generating command templates in $CMD_DIR ..."
  for py in "$SRC_DIR"/*.py; do
    [ -f "$py" ] || continue
    base=$(basename "$py" .py)
    cmdfile="$CMD_DIR/$base.sh"
    if [[ -f "$cmdfile" && ${FORCE:-0} -eq 0 ]]; then
      echo "  Skipping existing template: $cmdfile (use --force to overwrite)"
      continue
    fi

    # try extract example command from top docstring using a small Python helper
    tmpf=$(mktemp)
    python3 - "$py" > "$tmpf" <<'PY' || true
import sys, ast
path = sys.argv[1]
out = ''
try:
    with open(path, 'r', encoding='utf-8') as f:
        src = f.read()
    mod = ast.parse(src)
    doc = ast.get_docstring(mod) or ''
    for l in doc.splitlines():
        if 'python' in l.lower():
            out = l.strip()
            break
except Exception:
    pass
print(out)
PY
    example_cmd=$(cat "$tmpf" || true)
    rm -f "$tmpf" || true

    cat > "$cmdfile" <<-'SCRIPT'
#!/usr/bin/env bash
set -euo pipefail
# Edit this command to add per-file specific arguments. Use $COMMON_ARGS for shared args.
# Example (if available) is included below as a commented line.

COMMON_ARGS=${COMMON_ARGS:-""}

SCRIPT

    if [[ -n "$example_cmd" ]]; then
      # write commented example
      echo "# Example from $py:" >> "$cmdfile"
      echo "# $example_cmd" >> "$cmdfile"
    else
      echo "# No example command found in $py; please edit and provide one." >> "$cmdfile"
    fi

    # default runnable command: prefer the example command extracted from docstring
    module_path="Data_generation.sampler.question_generator.$base"
    if [[ -n "$example_cmd" ]]; then
      # use the example command (from docstring) as the default runnable line so required flags are included
      echo "$example_cmd \$COMMON_ARGS" >> "$cmdfile"
    else
      echo "python -m $module_path \$COMMON_ARGS" >> "$cmdfile"
    fi
    chmod +x "$cmdfile"
    echo "  Wrote template: $cmdfile"
  done
  echo "Templates generation done. Edit files under $CMD_DIR to configure per-file args." 
fi

if [[ $RUN -eq 1 ]]; then
  export COMMON_ARGS
  echo "Running all command templates in $CMD_DIR with COMMON_ARGS='$COMMON_ARGS'"
  # attempt to set PYTHONPATH to repository root so `python -m Data_generation...` works
  # find repo root by searching for setup.py or requirements.txt upward from HERE_DIR
  REPO_ROOT=""
  p="$HERE_DIR"
  for i in {1..6}; do
    if [[ -f "$p/setup.py" ]] || [[ -f "$p/requirements.txt" ]]; then
      REPO_ROOT="$p"
      break
    fi
    p=$(dirname "$p")
  done
  if [[ -n "$REPO_ROOT" ]]; then
    export PYTHONPATH="$REPO_ROOT:${PYTHONPATH:-}"
    echo "Exported PYTHONPATH=$REPO_ROOT"
  else
    echo "Warning: repo root not found; if python -m import fails, run with PYTHONPATH set to project root or edit templates to call the .py file directly."
  fi
  shopt -s nullglob
  files=("$CMD_DIR"/*.sh)
  if [[ ${#files[@]} -eq 0 ]]; then
    echo "No templates found in $CMD_DIR. Run with --generate first."; exit 1
  fi
  for f in "${files[@]}"; do
    echo
    echo "==== Running: $(basename "$f") ===="
    echo "Command contents:"
    sed -n '1,500p' "$f" | sed -n '1,500p'
    echo "---- executing ----"
    bash "$f"
    echo "==== Finished: $(basename "$f") ===="
  done
  echo "All done."
fi

exit 0
