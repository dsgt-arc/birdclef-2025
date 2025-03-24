import torch

if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        _ = torch.cuda.get_device_name(0)
    print(
        f"CUDA initialized early: available={torch.cuda.is_available()}, devices={device_count}"
    )
