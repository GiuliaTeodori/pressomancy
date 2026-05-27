#!/usr/bin/env bash
# make_rankfiles.sh
# Usage: make_rankfiles.sh <output_base_dir> <total_cores> <ranks_per_task> <pattern1> [pattern2 ...]
# Creates a subfolder under <output_base_dir> with rankfiles named rankfile_<pattern>
# Prints the absolute path to the created subfolder on stdout.
set -euo pipefail

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 4 ]]; then
  echo "Usage: $(basename "$0") <output_base_dir> <total_cores> <ranks_per_task> <pattern1> [pattern2 ...]" >&2
  exit 2
fi

out_base=$1; shift
total_cores=$1; shift
ranks_per_task=$1; shift
patterns=("$@")

# Basic validation
re='^[0-9]+$'
if ! [[ $total_cores =~ $re && $ranks_per_task =~ $re ]]; then
  echo "Error: <total_cores> and <ranks_per_task> must be integers." >&2
  exit 1
fi
if (( total_cores <= 0 || ranks_per_task <= 0 )); then
  echo "Error: values must be positive." >&2
  exit 1
fi
num_tasks=${#patterns[@]}
if (( num_tasks == 0 )); then
  echo "Error: provide at least one pattern (task)." >&2
  exit 1
fi
if (( total_cores % ranks_per_task != 0 )); then
  echo "Error: total_cores ($total_cores) is not divisible by ranks_per_task ($ranks_per_task)." >&2
  exit 1
fi
if (( ranks_per_task * num_tasks != total_cores )); then
  echo "Error: ranks_per_task ($ranks_per_task) * num_tasks ($num_tasks) != total_cores ($total_cores)." >&2
  echo "Hint: Either adjust PARALLEL_PARAMS count or the core/rank numbers." >&2
  exit 1
fi

# Build a deterministic subfolder name
joined_patterns=$(printf "%s_" "${patterns[@]}")
joined_patterns=${joined_patterns%_}
subdir="rankfiles"

mkdir -p "$out_base/$subdir"
out_dir=$(cd "$out_base/$subdir" && pwd)

# Generate one rankfile per pattern chunk
for i in "${!patterns[@]}"; do
  pattern=${patterns[$i]}
  start=$(( i * ranks_per_task ))
  file="$out_dir/rankfile_${pattern}"
  : > "$file"
  for (( r=0; r<ranks_per_task; r++ )); do
    slot=$(( start + r ))
    echo "rank $r=localhost slot=$slot" >> "$file"
  done
  echo "Wrote $file" >&2
done
