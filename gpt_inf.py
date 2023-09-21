import torch
import argparse
import time
import os
from gpt_language_model import GPTLanguageModel, decode

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
parser.add_argument("--insid", default=100, type=int, help="Instance-ID")
parser.add_argument("--intrathread", default=64, type=int, help="Number of threads running Inference")
args = parser.parse_args()

if args.device_type == "gpu" and torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Load model
config = model_configs[args.model]
model = GPTLanguageModel(vocab_size, config['n_embd'], config['n_head'], config['n_layer'], block_size, dropout).to(device)

os.environ["OMP_NUM_THREADS"] = str(args.intrathread)
torch.set_num_threads(args.intrathread)
#below way of loading loaded unnecessary keys.. and gave me error for inference:TBD
#model.load_state_dict(torch.load(config["checkpoint_path"]))


checkpoint = torch.load(config["checkpoint_path"])
model.load_state_dict(checkpoint["model_state_dict"])
# Inference
context = torch.zeros((args.bs, 1), dtype=torch.long, device=device)

start_time = time.time()

for _ in range(args.num_iter):
    print(decode(model.generate(context, args.maxnewtoken, block_size)[0].tolist()))

end_time = time.time()
print("inftime=%s  BS=%s ins=%s token=%s\n" %((end_time - start_time), args.bs, args.insid, args.maxnewtoken))
print(f"Time taken for {args.num_iter} inferences: {end_time - start_time:.2f} seconds")

