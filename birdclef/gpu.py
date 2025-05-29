import torch
import tensorflow as tf


def is_gpu_enabled():
    """Check if either PyTorch or TensorFlow detects an available GPU."""
    pytorch_gpu = torch.cuda.is_available()
    tensorflow_gpu = len(tf.config.list_physical_devices("GPU")) > 0
    return pytorch_gpu and tensorflow_gpu


if is_gpu_enabled():
    print("GPU is enabled via PyTorch or TensorFlow.")
else:
    print("No GPU enabled via PyTorch or TensorFlow.")
