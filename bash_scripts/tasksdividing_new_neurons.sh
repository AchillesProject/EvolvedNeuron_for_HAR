#!/usr/bin/env bash

readonly CORE_IDLE_THRESHOLD=90;
readonly CORE_USED_THRESHOLD=25;
readonly TOP_LOOP_NUMBER=3;
readonly TOP_DELAY_NUMBER=2;
readonly SLEEP_COUNT=0.1; #in second
files_array=();
cpus_array=(8 9 10 11 12 13 14 15);
cpus_array_length=${#cpus_array[@]};
cpus_assigned_array=();
cpu_index=0;

python_scripts_path='../pythons/new_neurons/run/*.py'

for file in $python_scripts_path; do
    python_scripts+=($file);
done

python_scripts_no="${#python_scripts[@]}"

for python_script in "${python_scripts[@]}"; do
    echo $python_script;
    python_script_name=$(echo $python_script | cut -d '/' -f 5 | cut -d '.' -f 1);
    python_script_log="../pythons/new_neurons/logs/log_${python_script_name}.txt";
    echo "Running ${python_script_name} script and logging at ${python_script_log}.";
    `taskset -c "${cpu_index}-$((${cpu_index}+3))" python "${python_script}" &>"${python_script_log}" &!`;
    cpu_index=$(($cpu_index+3));
    echo "${cpu_index}"
done
