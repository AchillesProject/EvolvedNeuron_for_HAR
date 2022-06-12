#!/usr/bin/env bash

datasets=(3W Crop DoublePendulum ECG5000 FordB InsectWingbeat LSST WISDM);
loop_no=30;
python_script="../pythons/new_neurons/run/best21_v4_8sets.py";
cpu_index=0;

for dataset in "${datasets[@]}"; do
    echo $dataset;
    for ((i=1;i<=loop_no;i++)); do
        log_dir="../pythons/new_neurons/logs/log_best21_v4_${dataset}_${i}.txt";
        echo "${log_dir}-${cpu_index}";
        echo "taskset -c ${cpu_index} python3 ${python_script} ${dataset} ${i} &>${log_dir} &!";
        `taskset -c "$cpu_index"  python3 "$python_script" "$dataset" "$i" &>"$log_dir" &!`;
        ((cpu_index += 1));
    done
done
