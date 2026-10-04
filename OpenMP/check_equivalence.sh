#!/bin/bash
# Check that the OpenMP K-means gives the same clustering as the
# sequential version: same cluster sizes, centroids equal to within
# 1e-3 (the programs print 4 decimals; reduction order may differ).
#
# Usage: ./check_equivalence.sh [K] [data file]
set -euo pipefail
cd "$(dirname "$0")"

K=${1:-20}
DATA=${2:-data_20k.csv}
THREADS=${THREADS:-"1 2 4 8"}

make -s all

centroids() { grep "^Centroid" | sed -E 's/Centroid ([0-9]+): \(([^,]+), ([^)]+)\), Size: ([0-9]+)/\1 \2 \3 \4/'; }

ref=$(./kmeans_sequential "$DATA" "$K" | centroids)
fail=0
for sched in static dynamic; do
    for t in $THREADS; do
        got=$(./kmeans_parallel "$DATA" "$K" "$t" "$sched" | centroids)
        if paste -d' ' <(echo "$ref") <(echo "$got") | awk '
            { if ($1 != $5 || $4 != $8) bad = 1
              dx = $2 - $6; dy = $3 - $7
              if (dx < 0) dx = -dx; if (dy < 0) dy = -dy
              if (dx > 1e-3 || dy > 1e-3) bad = 1 }
            END { exit bad }'; then
            echo "PASS  schedule=$sched threads=$t"
        else
            echo "FAIL  schedule=$sched threads=$t"; fail=1
        fi
    done
done
exit $fail
