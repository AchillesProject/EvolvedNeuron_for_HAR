#!/usr/bin/env bash

readonly CORE_IDLE_THRESHOLD=80;
files_array=();
cpus_array=(8 9 10 11 12 13 14 15);

for file in ../Version9.128timesteps/*; do
    files_array+=($file);
done

echo "Total files: ${#files_array[@]}";

for file in "${files_array[@]}";
do
    echo "Current file: $file";
    flag=0;
    while : 
    do
        if [ "$flag" -eq "1" ]; then
            break
        else 
            for cpu in "${cpus_array[@]}";
            do
                cpu_str="Cpu$cpu"
                core_idle_values_8=$(top -b -n 2 | grep "$cpu_str" | awk '{print $8}' | cut -f 2 -d ',' | cut -f 1 -d '.');
                readarray -t core_idle_lines_8 <<< "$core_idle_values_8"
                core_idle_values_9=$(top -b -n 2 | grep "$cpu_str" | awk '{print $9}' | cut -f 2 -d ',' | cut -f 1 -d '.');
                readarray -t core_idle_lines_9 <<< "$core_idle_values_9"
                if [ "${#core_idle_lines_8[@]}" -eq 2 ];
                then
                    core_idle=${core_idle_lines_8[-1]}
                else
                    core_idle=${core_idle_lines_9[-1]}
                fi
                echo "$cpu_str : $core_idle with threshold $CORE_IDLE_THRESHOLD";
                if [ $core_idle -gt $CORE_IDLE_THRESHOLD ];
                then
                    `taskset -c "$cpu" ./whileloop.sh &>/dev/null &!`;
                    echo "Done assign $cpu_str to file $file";
                    flag=1;
                    break
                fi
            done
        fi
    done
done
