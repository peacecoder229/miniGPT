#!/bin/bash
#for m in "10m"; do 
for m in "10m" "80m" "450m"; do 
    #for ipex_option in "--ipex"; do 
    for ipex_option in ""; do 
        if [ "$ipex_option" == "--ipex" ]; then
            tag="deltaipex"
        else
            tag="cudamemprofilev2"
            #tag="7Bcudaonly"
            #tag="cuda"
        fi
       ./intel_ipx_llama2_sweep.sh --model "$m" --tag $tag --newtoken [128,256] $ipex_option;
        #./intel_ipx_llama2_sweep.sh --model "$m" --tag $tag --newtoken [128,256] $ipex_option --prof;
       #sleep 5
	sleep 120	
       python check_pt_gpu.py
       sleep 1
    done; 
done

