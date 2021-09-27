#!/usr/bin/env bash

readonly CORE_IDLE_THRESHOLD=95;
readonly TOP_LOOP_NUMBER=10;
readonly TOP_DELAY_NUMBER=10;
readonly SLEEP_COUNT=0.1; #in second
files_array=();
cpus_array=(8 9 10 11 12 13 14 15);
cpus_array_length=${#cpus_array[@]};
cpu_index=0;

for file in ../Version9.128timesteps/*; do
    files_array+=($file);
done

starttime_date=`date`;
starttime_second=`data +%s`;
echo "Starting Hyperparameter Tunning with ${#files_array[@]} total file at $starttime_date ($starttime_second).";

for file in "${files_array[@]}";
do
    #echo "Current file: $file";
    ni_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 2 | cut -d '=' -f 2);
    no_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 3 | cut -d '=' -f 2);
    mc_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 4 | cut -d '=' -f 2);
    timestep_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 5 | cut -d 's' -f 2);
    file_no=$(echo $file | cut -d '/' -f 3 | cut -d '.' -f 7);
    filename="$ni_no""_""$no_no""_""$mc_no""_""$timestep_no""_""$file_no";
    flag=0;
    while : 
    do
        if [ "$flag" -eq "1" ]; then
            break
        else 
            while :
            do
                cpu=${cpus_array[$cpu_index]};
                cpu_str="Cpu$cpu"
                if [[ $cpu_index -ge $(($cpus_array_length - 1)) ]]; then
                    cpu_index=0;
                else
                    ((cpu_index += 1));
                fi

                core_idle_values=$(top -b -n "$TOP_LOOP_NUMBER" -d "$TOP_DELAY_NUMBER" | grep "$cpu_str" | awk '{print $8 $9}' | cut -f 2 -d ',' | cut -f 1 -d '.');
                readarray -t core_idle_lines <<< "$core_idle_values"

                total=0;
                sum=0;
                for i in "${core_idle_lines[@]}"; do
                    if [[ ! -z "$i" ]]; then
                        sum=$(($sum + $i));
                        ((total++));
                    fi
                done
                core_idle=$((sum/total));
                #echo "$cpu_str : $core_idle with threshold $CORE_IDLE_THRESHOLD";

                if [ $core_idle -gt $CORE_IDLE_THRESHOLD ];
                then
                    log_dir="../logs/processing/log_${filename}.txt"
                    #`taskset -c "$cpu" ./whileloop.sh &>/dev/null &!`;
                    #`taskset -c "$cpu"  python ./Hyperband_1Datasets.py mse "$file"&>/dev/null &!`;
                    `taskset -c "$cpu"  python ./Hyperband_1Datasets.py mse "$file" &>"$log_dir" &!`;
                    echo "Assigning $(($cpu_str+1)) to file number $filename";
                    flag=1;
                    break
                fi
            done
            sleep $SLEEP_COUNT;
        fi
    done
done

endtime_date=`date`;
endtime_second=`data +%s`;
echo "Finish Hyperparameter Tunning with ${#files_array[@]} total file at $endtime_date ($endtime_second) in $(($endtime_second - $starttime_second)).";
