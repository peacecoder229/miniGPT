import torch
import argparse
import time
import os
import sys
from gpt_language_model import GPTLanguageModel, decode
from torch.profiler import tensorboard_trace_handler
#from torch.utils.tensorboard import SummaryWriter

log_dir = "/nvm1/profiling/"
#writer = SummaryWriter(log_dir)
handler = tensorboard_trace_handler(log_dir, "trace_result")
# Config dictionary
model_configs = {
    "10m": {
        "n_embd": 384,
        "n_head": 6,
        "n_layer": 6,
        "checkpoint_path": "/nvm1/checkpoints/10m/checkpoint_4999.pth"
    },
    "80m": {
        "n_embd": 768,
        "n_head": 12,
        "n_layer": 12,
        "checkpoint_path": "/nvm1/checkpoints/80m/checkpoint_4999.pth"
    },
    "150m": {
        "n_embd": 960,
        "n_head": 16,
        "n_layer": 16,
        "checkpoint_path": "/nvm1/checkpoints/150m/checkpoint_4999.pth"
    },
    "450m": {
        "n_embd": 1024,
        "n_head": 16,
        "n_layer": 40,
        "checkpoint_path": "/nvm1/checkpoints/450m/checkpoint_4999.pth"
    },
}

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
chars = sorted(list(set(text)))
vocab_size = len(chars)
block_size = 256
dropout = 0.2
# Argument parser
parser = argparse.ArgumentParser(description='GPT Inference Script')
parser.add_argument('--model', type=str, required=True, choices=["10m", "80m", "150m", "450m"],
                    help='Which model to use: 10m, 80m, 150m, 450m')
parser.add_argument('--bs', type=int, default=1, help='Batch size for inference')
parser.add_argument('--maxnewtoken', type=int, default=500, help='Maximum new tokens to generate')
parser.add_argument('--num_iter', type=int, default=1, help='Number of iterations for inference timing')
parser.add_argument("--device_type", type=str, choices=['cpu', 'gpu'], required=True, help="Device type for running the model")
parser.add_argument("--ipex", action="store_true")
parser.add_argument("--prof", action="store_true")
parser.add_argument("--printmodeldata", action="store_true")
parser.add_argument("--insid", default=100, type=int, help="Instance-ID")
parser.add_argument("--intrathread", default=64, type=int, help="Number of threads running Inference")
args = parser.parse_args()

if args.device_type == "gpu" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Load model
config = model_configs[args.model]
model = GPTLanguageModel(vocab_size, config['n_embd'], config['n_head'], config['n_layer'], block_size, dropout)
model = model.eval().to(device)

if args.ipex:
    import intel_extension_for_pytorch as ipex
    model = ipex.optimize(model.eval(), dtype=torch.bfloat16)
    model = model.to(device)
else:
    print("Executing non Ipex path\n")

os.environ["OMP_NUM_THREADS"] = str(args.intrathread)
torch.set_num_threads(args.intrathread)
#below way of loading loaded unnecessary keys.. and gave me error for inference:TBD
#model.load_state_dict(torch.load(config["checkpoint_path"]))



try:
    checkpoint = torch.load(config["checkpoint_path"])
except RuntimeError as e:
    print(f"Couldn't load model normally, error: {str(e)}")
    print("Loading model to CPU instead")
    checkpoint = torch.load(config["checkpoint_path"], map_location=torch.device('cpu'))

model.load_state_dict(checkpoint["model_state_dict"])

if args.printmodeldata:
    print("print model state dict sizes #############################################################\n\n")
    model_state_dict = model.state_dict()
    size_in_bytes = sys.getsizeof(model_state_dict)
    print("Model State Dict Size =%sMB\n" %(size_in_bytes / (1024 * 1024)))
    total_size_bytes = 0
    #Alternatively, print only keys to get an overview
    print("Keys in model_state_dict:")
    for key in model_state_dict.keys():
        print(key)

             # Or, print keys along with the shape of the corresponding tensor
    print("Shapes of tensors in model_state_dict:")
    for key, tensor in model_state_dict.items():
        print(f"{key}: {tensor.shape}")
        element_size = tensor.element_size()  # size of each element in bytes
        num_elements = tensor.numel()  # number of elements in the tensor
        tensor_size_bytes = element_size * num_elements  # size of tensor in bytes
        total_size_bytes += tensor_size_bytes  # update the total size

        print(f"{key}: shape={tensor.shape}, size={tensor_size_bytes} bytes")

    total_size_megabytes = total_size_bytes / (1024 * 1024)
    print(f"Total size of all tensors: {total_size_megabytes:.2f} MB")



# Inference

if device.lower() == "cpu" and args.ipex:
    print("Executing Ipex path\n")
    with torch.inference_mode(), torch.no_grad(), torch.autocast(device_type="cpu", enabled=True, dtype=getattr(torch, "bfloat16")):
        context = torch.zeros((args.bs, 1), dtype=torch.long, device=device)
        start_time = time.time()
        for _ in range(args.num_iter):
            print(decode(model.generate(context, args.maxnewtoken, block_size)[0].tolist()))

        end_time = time.time()
        print("inftime=%s  BS=%s ins=%s token=%s\n" %((end_time - start_time), args.bs, args.insid, args.maxnewtoken))

elif args.prof:
    with torch.profiler.profile(with_stack=True, profile_memory=True,
                            #schedule=torch.profiler.schedule(wait=1, warmup=1, active=3),
                            #record_shapes=True,  on_trace_ready=handler) as prof: #for genetaing traces with trace_handler.
                            record_shapes=True) as prof:
        context = torch.zeros((args.bs, 1), dtype=torch.long, device=device)
        start_time = time.time()
        for _ in range(args.num_iter):
            print(decode(model.generate(context, args.maxnewtoken, block_size)[0].tolist()))
            prof.step()

        end_time = time.time()
        #print("inftime=%s  BS=%s ins=%s token=%s\n" %((end_time - start_time), args.bs, args.insid, args.maxnewtoken))
    #handler.flush()
    #handler.close()
    #writer.close()
    avg_events = prof.key_averages()
    #sorted_events = sorted(avg_events, key=lambda event: event.cpu_time_total, reverse=True)
    #sorted_events = sorted(avg_events, key=lambda event: (-event.self_cpu_memory_usage, event.count, -event.self_cpu_time_total) )
    sorted_events = sorted(avg_events, key=lambda event: -event.self_cpu_time_total )
    total_cpu_time = sum([event.self_cpu_time_total for event in sorted_events])
    cumulative_cpu_time = 0
    significant_events = []
    total_avg_memory_usage_significant = 0
    total_memory_usage_all_events = 0 
    #print("Event Name,CPU Time Total,Self CPU Time Total,Total CPU Memory Usage,Self CPU Memory Usage,Call Count,Stack\n")
    for event in sorted_events:
        event_name = event.key if hasattr(event, 'key') else 'Unknown'
        stack_trace = '->'.join(event.stack) if hasattr(event, 'stack') else 'Unknown'
        cumulative_cpu_time += event.self_cpu_time_total
        avg_memory_usage_per_call = (event.self_cpu_memory_usage / event.count)/1e6 if event.count > 0 else 0
        total_memory_usage_all_events += avg_memory_usage_per_call  # Accumulating the total average memory usage.
        #print(f"{event_name},{event.cpu_time_total},{event.self_cpu_time_total},{event.self_cpu_memory_usage},{event.cpu_memory_usage},{event.count},{stack_trace}")
        if cumulative_cpu_time < 0.95 * total_cpu_time:
            total_avg_memory_usage_significant += avg_memory_usage_per_call
            significant_events.append(event)


    #print("\nSignificant Events contributing to > 95% of total CPU time:")
    for event in significant_events:
        event_name = event.key if hasattr(event, 'key') else 'Unknown'
        #print(f"{event_name},{event.cpu_time_total},{event.self_cpu_time_total},{event.self_cpu_memory_usage},{event.cpu_memory_usage},{event.count},{avg_memory_usage_per_call},{stack_trace}")
    #print("significant_ops_memFP=%s\ntotal_ops_memFP=%s\n" %(total_avg_memory_usage_significant, total_memory_usage_all_events))

    print("inftime=%s  BS=%s ins=%s token=%s sigmem=%s totmem=%s\n" %((end_time - start_time), args.bs, args.insid, args.maxnewtoken, total_avg_memory_usage_significant, total_memory_usage_all_events))
else:
    context = torch.zeros((args.bs, 1), dtype=torch.long, device=device)
    #context = torch.zeros((1, 1), dtype=torch.long, device=device)

    start_time = time.time()

    for _ in range(args.num_iter):
        print(decode(model.generate(context, args.maxnewtoken, block_size)[0].tolist()))
        '''
        #below generates BS number of lines with maxtokenlength. starts from  context with all elements 0
        #
        generated_output = model.generate(context, args.maxnewtoken, block_size)
        for idx in range(generated_output.shape[0]):  # Iterate over the batch dimension
            decoded_text = decode(generated_output[idx].tolist())
            print(decoded_text)
        '''

    end_time = time.time()
    print("inftime=%s  BS=%s ins=%s token=%s\n" %((end_time - start_time), args.bs, args.insid, args.maxnewtoken))

