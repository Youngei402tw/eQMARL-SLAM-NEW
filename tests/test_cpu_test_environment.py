"""Regression coverage for CPU-only local tests."""

import os

import tensorflow as tf


def test_pytest_hides_gpus_before_tensorflow_imports():
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    assert tf.config.list_physical_devices("GPU") == []
