import subprocess
import time
import sys
def get_gpu_stats():
    """
    Get the current GPU memory usage and compute utilization.

    Returns two dictionaries:
        - GPU id to memory used by each GPU
        - GPU id to compute utilization of each GPU
    """

    memory_result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in memory_result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))

    utilization_result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=utilization.gpu',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_utilization = [int(x) for x in utilization_result.strip().split('\n')]
    gpu_utilization_map = dict(zip(range(len(gpu_utilization)), gpu_utilization))

    return gpu_memory_map, gpu_utilization_map

# Call the function every second to print the GPU memory usage and compute utilization
print("memused,compute\n")
while True:
    dly = int(sys.argv[1])
    mem_map, util_map = get_gpu_stats()
    #print("Memory Map:", mem_map)
    #print("Compute Utilization Map:", util_map)
    print("%s,%s\n" %(mem_map[0]/1e3, util_map[0]))
    time.sleep(dly)

