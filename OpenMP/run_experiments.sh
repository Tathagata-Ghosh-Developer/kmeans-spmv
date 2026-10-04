#!/bin/bash
K=${1:-20}
REPEAT=${REPEAT:-5}  # runs averaged per configuration (results/ used REPEAT=30)
THREADS_LIST=${THREADS_LIST:-"4 8 16 32"}
SCHEDULES="static dynamic"
DATAFILE="data_20k.csv"
CLUSTER_CSV="cluster_outputs.csv"

if ! [ -x "./kmeans_sequential" ] || ! [ -x "./kmeans_parallel" ]; then
    echo "Programs not built. Building now..."
    make || { echo "Build failed! Exiting."; exit 1; }
fi

# Append one CSV row per cluster from a program's output.
# Input lines look like: "Centroid 3: (-1166.2891, 353.4936), Size: 1012"
# With -F '[():,]' the fields are $1="Centroid 3", $3=x, $4=y, $7=size.
append_clusters() {
    echo "$1" | grep -E "^Centroid" | awk -F '[():,]' -v mode="$2" -v threads="$3" '{
        sub(/^Centroid /, "", $1)
        gsub(/ /, "", $3); gsub(/ /, "", $4); gsub(/ /, "", $7)
        printf("%s,%s,%s,%s,%s,%s\n", mode, threads, $1, $7, $3, $4)
    }' >> "$CLUSTER_CSV"
}

rm -f "$CLUSTER_CSV"
echo "Mode,Threads,ClusterIndex,PointCount,CentroidX,CentroidY" > "$CLUSTER_CSV"

# Sequential run
echo "Running sequential version $REPEAT times for K=$K"
seq_times=""
for (( run=1; run<=REPEAT; run++ )); do
    output=$(./kmeans_sequential "$DATAFILE" $K)
    time_val=$(echo "$output" | grep "Time" | awk '{print $2}')
    seq_times+="$time_val\n"
    echo "  Run $run: ${time_val} ms"

    if [ $run -eq $REPEAT ]; then
        append_clusters "$output" Sequential 1
    fi

done
avg_seq=$(echo -e "$seq_times" | awk 'NF>0 {sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count}')
echo "Average Sequential Time: ${avg_seq} ms"
echo ""

# Parallel runs
echo "Benchmarking parallel version for various thread counts and schedules (each averaged over $REPEAT runs)"
echo "Threads,Static_avg_ms,Dynamic_avg_ms"
for t in $THREADS_LIST; do
    times_static=""
    times_dynamic=""

    for (( run=1; run<=REPEAT; run++ )); do
        out_static=$(./kmeans_parallel "$DATAFILE" $K $t static)
        time_static=$(echo "$out_static" | grep "Time" | awk '{print $2}')
        times_static+="$time_static\n"
        if [ $run -eq $REPEAT ]; then
            append_clusters "$out_static" Static "$t"
        fi
    done

    for (( run=1; run<=REPEAT; run++ )); do
        out_dynamic=$(./kmeans_parallel "$DATAFILE" $K $t dynamic)
        time_dynamic=$(echo "$out_dynamic" | grep "Time" | awk '{print $2}')
        times_dynamic+="$time_dynamic\n"
        if [ $run -eq $REPEAT ]; then
            append_clusters "$out_dynamic" Dynamic "$t"
        fi
    done

    static_avg=$(echo -e "$times_static" | awk 'NF>0 {sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count}')
    dynamic_avg=$(echo -e "$times_dynamic" | awk 'NF>0 {sum+=$1; count++} END {if(count>0) printf "%.4f", sum/count}')

    echo "$t,${static_avg},${dynamic_avg}"
done

echo "All cluster outputs saved to $CLUSTER_CSV"
