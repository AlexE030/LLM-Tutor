import torch

print(f"CUDA verfügbar: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Anzahl GPUs: {torch.cuda.device_count()}")
    print(f"Aktuelle GPU: {torch.cuda.current_device()}")
    print(f"GPU-Name: {torch.cuda.get_device_name(0)}")