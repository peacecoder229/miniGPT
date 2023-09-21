#!/bin/bash
for m in "7b"; do 
    #for ipex_option in "--ipex" ""; do 
    for ipex_option in ""; do 
        if [ "$ipex_option" == "--ipex" ]; then
            tag="deltaipex"
        else
            #tag="nonipex"
            #tag="7Bcudaonly"
            tag="cuda_with_intellib"
        fi
        ./intel_ipx_llama2_sweep.sh --model "$m" --tag $tag --newtoken [32] $ipex_option;
       sleep 5	
       python check_pt_gpu.py
       sleep 1
    done; 
done

