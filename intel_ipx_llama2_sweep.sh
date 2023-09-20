#!/bin/bash

# Define arrays of values you want to test
#cpupower frequency-set --governor performance
sleep 1
intrathread_values=(48)
device_type_values=("gpu")
#ins_values=({0..2} {0..4} {0..6})
#declare -A ins_values=(["64"]=2 ) 
#declare -A ins_values=(["64"]=2 ["32"]=4 ["16"]=8) 
declare -A ins_values=(["64"]=2) 
#ins_values=(4 8)
#bs_values=(64)
#bs_values=(1 16 32 64 128 256)
bs_values=(36)
rm -f tmp_util
#declare -a pids
i=0  #track odd instnaces 
k=0  #track eeven istancs
#echo "intrathread,device_type,ins,bs,{pids[-1]},mininfT,maxinfT,avginfT,gpuutilavg,gpuutilmax,memutilmax" >> result_sum.txt
# Loop over all combinations of values
#for intrathread in ${intrathread_values[@]}; do
#README: for each device type  it will spanwn all instances in parallel as definedd by maxinstnaces and collect data.

EMON=false
IPEX=false
PROF=false
TAG=""
TOKENS=()

while [[ $# -gt 0 ]]; do
    arg="$1"
    case $arg in
        --prof)
        PROF=true
        shift # Remove --emon from processing
	;;
        --ipex)
        IPEX=true
        shift # Remove --emon from processing
	;;
        --emon)
        EMON=true
        shift # Remove --emon from processing
	;;
	--tag)
	shift # Move to next argument which should be the value for --tag
	TAG=$1
	shift # 
        ;;
	--model)
	shift # Move to next argument which should be the value for --tag
	model=$1
	shift # 
        ;;
	--newtoken)
	echo "Entered --newtoken case"  # Debug prin
        shift
	echo "Argument after shift: $1"  # Debug print
	IFS=',' read -ra TOKENS <<< "${1//[\[\]]/}"
    	echo "Collected tokens after --newtokens: ${TOKENS[@]}"  # Debug print
	shift
        ;;
        *)
        # other arguments can be processed here if necessary
        shift
        ;;
    esac
done

port=29500
#for token in "8" "16" "24" "32" "64"; do
echo "Collected tokens: ${TOKENS[@]}"

sleep 2
#set -x

for token in "${TOKENS[@]}" ; do
  for device_type in ${device_type_values[@]}; do
    echo "$device_type"
    for bs in ${bs_values[@]}; do
      echo 3 > /proc/sys/vm/drop_caches && swapoff -a && swapon -a && printf '\n%s\n' 'Ram-cache and Swap Cleared'
      #python3 check_pt_gpu.py
      #for maxins in ${ins_values[@]}; do
      for intrathread in "${!ins_values[@]}"; do
	maxins=${ins_values[$intrathread]}
	k=0
	i=0
	newdir="log_dev${device_type}_ins${maxins}_BS${bs}"
	mkdir -p "Log/${newdir}"
	finalcmd=""
	for ins in $(seq 0 $(( maxins-1 )) ); do

        # Calculate d, s, and e values
		tmpfile="Log/res_${ins}.txt"
		rm -f $tmpfile
		port=$(( port+1 ))
		d=$(( ins*intrathread ))

		if [ "$device_type" = "cpu" ]; then
		  if (( (ins+1) % 2 == 0 )); then  # even ins
		    d=$(( k*intrathread ))
		    s=$(( 64+d ))
		    k=$(( k+1 ))
		  else                             # odd ins
		    d=$(( i*intrathread ))
		    s=$d
		    i=$(( i+1 ))
		  fi
		else
		  d=$(( ins*intrathread ))
		  s=$(( 64+d ))
		fi

		e=$(( s+intrathread-1 ))

		#!/bin/bash

		# Assuming PROF and IPEX can be either "true" or "false"

		if [ "$PROF" == "true" ] && [ "$IPEX" == "true" ]; then
			echo "Both PROF and IPEX are true."

			cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-${model}-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --ipex --jit --prof"
		elif [ "$IPEX" == "true" ]; then
			echo "Only IPEX is true."
			cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-${model}-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --ipex --jit"
		elif [ "$PROF" == "true" ]; then
			echo "Only PROF is true."
			cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-${model}-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --prof"

		else
			echo "Default case."
			#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-${model}-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token"
			#cmd="numactl -C ${s}-${e} python gpt_inf.py --device $device_type  --batchsize $bs --insid $ins --threads $intrathread --maxtokens $token --model ${model}"
			#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-7b --tokenizer_path tokenizer.model --max_seq_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
			#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-${model} --tokenizer_path tokenizer.model --max_gen_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
			#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins --model_type ${model} --ckpt_dir llama-2-${model} --tokenizer_path tokenizer.model --max_gen_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
			cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-${model} --tokenizer_path tokenizer.model --max_gen_len $token --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
		fi




		# Run your command and get the PID of the process
		#model is Llama-2-7b-hf or Llama-2-13b-hf
		#cmd="numactl -C ${s}-${e} python3 run_textgen.py --insid $ins  --ckpt_dir llama-2-7b --tokenizer_path tokenizer.model --max_seq_len 64 --max_batch_size $bs --device $device_type --promptfile prompt.txt --intrathread $intrathread"
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-7b-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --ipex --jit --accuracy-only"
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-${model}-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token --ipex --jit"
		#cmd="OMP_NUM_THREADS=$intrathread numactl -C ${s}-${e}  python run_generation.py --insid $ins --benchmark -m  meta-llama/Llama-2-7b-hf --dtype bfloat16 --num-iter 1 --batch-size $bs --input-tokens 612  --max_new_tokens $token"
		echo $cmd
		if [ $ins -eq $(( maxins-1 )) ]; then
			finalcmd+="$cmd > \"$tmpfile\""
		else
			finalcmd+="$cmd > \"$tmpfile\" &   "
		fi

		# Save the PID of the last background job (the one we just launched)

        # Echo values to the results file
        #echo "$intrathread,$device_type,$ins,$bs,${pids[-1]}" >> result_sum.txt
      done   #for ins in 
      echo $finalcmd
      rm -f /tmp/cpumemsize_tmp.txt
      if [ "$device_type" = "gpu" ]; then
	      python3 -u  print_gpu_util_stats.py 1 > tmp_util &
      else
	      echo "skipping GPU Util"
      fi
      if $EMON; then
	      tmc -T all -x rdas -i "NA" -D "${model}_BS${bs}_T${intrathread}_TK${token}${TAG}" -n -u -c "$finalcmd"
      else

	      #( eval "$finalcmd" ) &


	      (
		  while : ; do
		    cpumemsize=$(ps aux | grep python | awk '{print $2}' | xargs -I{} pmap -x {} | awk '/total/ {print $4}' | awk '{sum += $1} END {print sum}')
		    echo $cpumemsize >> /tmp/cpumemsize_tmp.txt
		    sleep 2
		  done
	       ) &

		bg_pid=$!
	
	       eval "$finalcmd" 


      fi

      kill $bg_pid

      maverage=$(awk '{sum+=$1} END {if(NR > 0) print sum/NR; else print 0}' /tmp/cpumemsize_tmp.txt)
      mmax_value=$(awk 'BEGIN {max = 0} {if ($1>max) max=$1} END {print max}' /tmp/cpumemsize_tmp.txt)

      #cpumemsize=$(cat /tmp/cpumemsize_tmp.txt | tail -n 1)
      echo "CPU used memory $maverage $mmax_value KB"
      #cpumemsize=$( ps aux | grep python | awk '{print $2}' | xargs -I{} pmap -x {} | awk '/total/ {print $4}' | awk '{sum += $1} END {print sum}')
      #eval "$finalcmd"
      echo "##################################Run Complete ##########################################" 
      if [ "$device_type" = "gpu" ]; then
	      pkill -f print_gpu_util_stats.py
	      gpuutilavg=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep mean | awk '{print $3}')
	      gpuutilmax=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep Max | awk '{print $3}')
	      memutilavg=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep mean | awk '{print $2}')
	      memutilmax=$(./generate_min_max_avg_sum.py --inputfile=tmp_util | grep Max | awk '{print $2}')
      else
	      echo "No GPU"
      fi

      sleep 15

      res=""
      for ins in $(seq 0 $(( maxins-1 )) ); do
	       tmpfile="Log/res_${ins}.txt"
	       #res+='$(grep "^inftime=" "$tmpfile")'
	       res+="$(grep "^inftime=" "$tmpfile"),"

      done

      res+=", $TAG"

      touch cpulog_${TAG}
      touch gpulog_${TAG}
      touch gpuutil_${TAG}

      if [ "$device_type" = "cpu" ]; then
	      inscnt=$(( ins+1 ))
	      echo "$maverage,$mmax_value,$model,$intrathread,$device_type,$inscnt,$bs,$res" >> cpulog_${TAG}
      else
	      inscnt=$(( ins+1 ))
	      echo "$maverage,$mmax_value,$model,$intrathread,$device_type,$inscnt,$bs,$res" >> gpulog_${TAG}
	      echo "$model,$intrathread,$device_type,$inscnt,$bs,$gpuutilavg,$gpuutilmax,$memutilavg,$memutilmax,$res" >> gpuutil_${TAG}
      fi
      #Using -u immediatey writes stdout to file

      # Wait for all jobs to finish before moving to the next bs value

      #pkill -f print_gpu_util_stats.py
     
     sleep 2
     #echo "$intrathread,$device_type,$ins,$bs,${pids[-1]},$mininfT,$maxinfT,$avginfT,$gpuutilavg,$gpuutilmax,$memutilavg,$memutilmax," >> result_sum.txt
      # Clear the PID array for the next loop iteration
    done  # for maxins

    done
  done
 done
#done

