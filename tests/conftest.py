"""Test-process environment controls."""

import os


# Tests exercise behavior, not accelerator throughput. Keep local GPUs available
# for unrelated training before TensorFlow is imported during test collection.
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
