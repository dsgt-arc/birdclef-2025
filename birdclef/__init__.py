import torch

# PyTorch CUDA detection
if torch.cuda.is_available():
    device_count = torch.cuda.device_count()
    if device_count > 0:
        _ = torch.cuda.get_device_name(0)
    print(
        f"CUDA initialized early: available={torch.cuda.is_available()}, devices={device_count}"
    )

# TensorFlow GPU detection
try:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            print(f"TensorFlow GPU found: {gpu}")
        print(f"TensorFlow initialized early: available GPUs={len(gpus)}")
    else:
        print("No TensorFlow GPU available")
except ImportError:
    print("TensorFlow not installed")
