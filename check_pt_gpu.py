import torch
import os
# Check PyTorch version
print("PyTorch version:")
print(torch.__version__)

print("Huggngface Transformer version=")
os.system("pip show transformers | grep Version")

print("PyTorch location being used is \n")
print(torch.__file__)

# Check if CUDA (GPU) is available
print("\nIs GPU available?")
print(torch.cuda.is_available())

# Check GPU details if GPU is present
if torch.cuda.is_available():
    print("\nGPU Details:")
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.get_device_name)
    print("Clearing GPU cache\n")
    torch.cuda.empty_cache()
    print("Cleared GPU cache\n")

    


