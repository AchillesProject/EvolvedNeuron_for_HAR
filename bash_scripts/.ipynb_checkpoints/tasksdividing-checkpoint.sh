#!/usr/bin/env bash

readonly CORE_IDLE_THRESHOLD=75;
readonly CORE_USED_THRESHOLD=25;
readonly TOP_LOOP_NUMBER=3;
readonly TOP_DELAY_NUMBER=2;
readonly SLEEP_COUNT=0.1; #in second
files_array=();
cpus_array=(8 9 10 11 12 13 14 15);
cpus_array_length=${#cpus_array[@]};
cpus_assigned_array=();
cpu_index=0;
lossmethod="mse"

for file in ../Version9.128timesteps/*; do
    files_array+=($file);
done

for i in "${cpus_array[@]}"; do
    cpus_assigned_array+=(0);
done

starttime_date=`date`;
starttime_second=`date +%s`;
echo "Starting Hyperparameter Tunning with ${#files_array[@]} total file on $starttime_date ($starttime_second).";

for file in "${files_array[@]}"; do
    ni_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 2 | cut -d '=' -f 2);
    no_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 3 | cut -d '=' -f 2);
    mc_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 4 | cut -d '=' -f 2);
    timestep_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 5 | cut -d 's' -f 2);
    version_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 6 | cut -d 'n' -f 2);
    file_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 7);
    filename="$lossmethod""_""$ni_no""_""$no_no""_""$mc_no""_""$timestep_no""_""$version_no""_""$file_no";
    flag=0;
    while : 
    do
        cpu=${cpus_array[$cpu_index]};
        cpu_str="Cpu$((cpu + 1))"
        core_idle_values=$(top -b -n "$TOP_LOOP_NUMBER" -d "$TOP_DELAY_NUMBER" | grep "$cpu_str" | awk '{print $3}' | cut -f 1 -d '.');
        readarray -t core_idle_lines <<< "$core_idle_values"

        total=0;
        sum=0;
        for i in "${core_idle_lines[@]}"; do
            if [[ ! -z "$i" ]]; then
                sum=$(($sum + $i));
            else
                sum=$(($sum + 0));
            fi
            ((total++));
        done
        core_idle=$((100 - sum/total));
        if [[ $core_idle -gt $CORE_IDLE_THRESHOLD ]]; then
            log_dir="../logs/processing/log_${filename}.txt"
            if [[ "${cpus_assigned_array[$cpu_index]}" == "0" ]]; then
                cpus_assigned_array[$cpu_index]=$filename;
                flag=1;
            else
                if [[ $(tail -1 ../logs/processing/log_"${cpus_assigned_array[$cpu_index]}".txt) == *"Complete"* ]] || [[ $(tail -1 ../logs/processing/log_"${cpus_assigned_array[$cpu_index]}".txt) == *"Error"* ]]; then                    
                    cpus_assigned_array[$cpu_index]=$filename;
                    flag=1;
                fi
            fi
        fi

        if [[ $cpu_index -ge $(($cpus_array_length - 1)) ]]; then
            cpu_index=0;
        else
            ((cpu_index += 1));
        fi

        if [[ "$flag" -eq "1" ]]; then
            `taskset -c "$cpu"  python ./Hyperband_1Datasets.py "$lossmethod" "$file" &>"$log_dir" &!`;
            echo "Assigning $cpu_str to file number $filename";
            break
        fi
    done
    sleep $SLEEP_COUNT;
done

# Check if all the processes has finished
idlecores=1;
while : 
do
    cpu=${cpus_array[$cpu_index]};
    core_idle_values=$(top -b -n "$TOP_LOOP_NUMBER" -d "$TOP_DELAY_NUMBER" | grep "$cpu_str" | awk '{print $3}' | cut -f 1 -d '.');
    readarray -t core_idle_lines <<< "$core_idle_values"

    total=0;
    sum=0;
    for i in "${core_idle_lines[@]}"; do
        if [[ ! -z "$i" ]]; then
            sum=$(($sum + $i));
        else
            sum=$(($sum + 0));
        fi
        ((total++));
    done
    core_idle=$((100 - sum/total));
    if [[ $core_idle -gt $CORE_IDLE_THRESHOLD ]]; then
        ((idlecores+=1))
    fi
    if (( $idlecores == $cpus_array_length)); then
        break
    fi
    if [[ $cpu_index -ge $(($cpus_array_length - 1)) ]]; then
        cpu_index=0;
    else
        ((cpu_index += 1));
    fi
done

endtime_date=`date`;
endtime_second=`date +%s`;
echo "Finish Hyperparameter Tunning with ${#files_array[@]} total file on $endtime_date ($endtime_second) in $(($endtime_second - $starttime_second)).";
