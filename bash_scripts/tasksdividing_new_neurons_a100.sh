#!/usr/bin/env bash

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
    `taskset -c "${cpu_index}-$((${cpu_index}+8))" python3 "${python_script}" &>"${python_script_log}" &!`;
    cpu_index=$(($cpu_index+8));
    echo "${cpu_index}"
done
