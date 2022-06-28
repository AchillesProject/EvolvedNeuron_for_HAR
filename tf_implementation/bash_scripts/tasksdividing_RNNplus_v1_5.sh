#!/usr/bin/env bash

readonly CORE_IDLE_THRESHOLD=90;
readonly CORE_USED_THRESHOLD=25;
readonly TOP_LOOP_NUMBER=3;
readonly TOP_DELAY_NUMBER=2;
readonly SLEEP_COUNT=0.1; #in second
readonly ISMOOREDATA="false";
files_array=();
models_array=("LSTM" "RNN_plus" "RNN");
LRSoptions_array=(true);
CMFoptions_array=(false);
cpus_array=(0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53 54 55 56 57 58 59 60 61 62 63 64 65 66 67 68 69 70 71 72 73 74 75 76 78 79 80 81 82 83 84 85 86 87 88 89 90 91 92 93 94 95 96 97 98 99 100 101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116 117 118 119 120 121 122 123 124 125 126 127);
cpus_array_length=${#cpus_array[@]};
cpus_assigned_array=();
cpu_index=0;

pythonscript="./RNNplus_v1_5_1Datasets.py";

for file in ../../Datasets/2_addingproblem/*; do
    files_array+=($file);
done

for i in "${cpus_array[@]}"; do
    cpus_assigned_array+=(0);
done

starttime_date=`date`;
starttime_second=`date +%s`;
echo "Starting Training with ${#files_array[@]} total file on $starttime_date ($starttime_second).";

for model in "${models_array[@]}"; do
	for LRSoption in "${LRSoptions_array[@]}"; do
		for CMFoption in "${CMFoptions_array[@]}"; do
			if [ "${LRSoption}" == "true" ]; then sLRS='wLRS'; else sLRS='wtLRS'; fi;
			if [ "${CMFoption}" == "true" ]; then sCMF='wCMF'; else sCMF='wtCMF'; fi;
			prefix="${model,,}""_""${sLRS}""_""${sCMF}""_tf";
			for file in "${files_array[@]}"; do
				if [ "${ISMOOREDATA}" == "true" ]; then                
					ni_no=$(echo $file | cut -d '/' -f 6 | cut -d '.' -f 2 | cut -d '=' -f 2);
					no_no=$(echo $file | cut -d '/' -f 6 | cut -d '.' -f 3 | cut -d '=' -f 2);
					mc_no=$(echo $file | cut -d '/' -f 6 | cut -d '.' -f 4 | cut -d '=' -f 2);
					timestep_no=$(echo $file | cut -d '/' -f 6 | cut -d '.' -f 5 | cut -d 's' -f 2);
					filename="$prefix""_""$ni_no""_""$no_no""_""$mc_no""_""$timestep_no";
				else             
					bs_no=$(echo $file | cut -d '/' -f 5 | cut -d '.' -f 2 | cut -d '=' -f 2);
					timestep_no=$(echo $file | cut -d '/' -f 5 | cut -d '.' -f 3 | cut -d '=' -f 2);
					filename="$prefix""_""$bs_no""_""$timestep_no";
                fi
				flag=0;
				while : 
				do
					cpu=${cpus_array[$cpu_index]};
					core_idle_values=$(top -b -n "$TOP_LOOP_NUMBER" -d "$TOP_DELAY_NUMBER" | grep "Cpu$((cpu)) " | awk -F: '{print $2}' | awk '{print $1}' | cut -f 1 -d '.');
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
						`taskset -c "$cpu"  python3 "$pythonscript" "$file" "$model" "$LRSoption" "$CMFoption" &>"$log_dir" &!`;
						echo "Assigning Cpu$((cpu)) to file $filename";
						break
					fi
				done
				sleep $SLEEP_COUNT;
			done
		done
	done
done
# Check if all the processes has finished
idlecores=0;
cpu_index=0;

while : 
do
    cpu=${cpus_array[$cpu_index]};
    core_idle_values=$(top -b -n "$TOP_LOOP_NUMBER" -d "$TOP_DELAY_NUMBER" | grep "Cpu$((cpu)) " | awk -F: '{print $2}' | awk '{print $1}' | cut -f 1 -d '.');
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
		echo "Break"
        break
    fi
    if [[ $cpu_index -ge $(($cpus_array_length - 1)) ]]; then
        cpu_index=0;
		idlecores=0;
    else
        ((cpu_index += 1));
    fi
done

endtime_date=`date`;
endtime_second=`date +%s`;
echo "Finish Training with ${#files_array[@]} total file on $endtime_date ($endtime_second) in $(($endtime_second - $starttime_second)).";
