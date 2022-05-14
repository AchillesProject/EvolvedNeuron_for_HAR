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

python_scripts = '../pythons/best21/*.py'

for file in ($python_scripts); do
    files_array+=($file);
done

for file in "${files_array[@]}"; do
    echo $file
done