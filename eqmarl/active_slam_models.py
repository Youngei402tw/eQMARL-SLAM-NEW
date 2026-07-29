"""MiniGrid-faithful model builders for the active-SLAM comparison."""

from __future__ import annotations

import functools as ft

import cirq
import tensorflow as tf
import tensorflow.keras as keras

from .layers import (
    HybridPartiteVariationalEncodingPQC,
    HybridVariationalEncodingPQC,
    RescaleWeighted,
)
from .ops import ParameterizedRotationLayer_RxRyRz


def _validate(observation_dim: int, n_agents: int, n_actions: int):
    if observation_dim <= 0:
        raise ValueError("observation_dim must be positive")
    if n_agents != 2:
        raise ValueError("the active-SLAM comparison requires exactly two agents")
    if n_actions <= 1:
        raise ValueError("n_actions must be at least two")


def generate_actor_classical(
    observation_dim: int,
    n_actions: int = 4,
    units: list[int] | None = None,
    activation: str = "relu",
    name: str = "active-slam-actor",
) -> keras.Model:
    """Shared classical actor used by all four MiniGrid-faithful methods."""
    _validate(observation_dim, 2, n_actions)
    units = units or [100]
    layers = [keras.Input(shape=(observation_dim,), dtype=tf.float32)]
    layers.extend(keras.layers.Dense(size, activation=activation) for size in units)
    layers.append(keras.layers.Dense(n_actions, activation="softmax", name="policy"))
    return keras.Sequential(layers, name=name)


def _joint_observable(qubits):
    return [ft.reduce(lambda left, right: left * right, [cirq.Z(qubit) for qubit in qubits])]


def _value_readout(observables, beta: float):
    return keras.Sequential(
        [
            RescaleWeighted(len(observables)),
            keras.layers.Lambda(lambda values: values * beta),
        ],
        name="observables-value",
    )


def generate_critic_eqmarl(
    observation_dim: int,
    n_agents: int = 2,
    n_actions: int = 4,
    d_qubits: int = 4,
    n_layers: int = 5,
    beta: float = 1.0,
    squash_activation: str = "arctan",
    nn_activation: str = "linear",
    trainable_w_enc: bool = False,
    input_entanglement_type: str = "psi+",
    name: str = "critic-eqmarl-psi+",
) -> keras.Model:
    """Partite critic matching the MiniGrid eQMARL architecture."""
    _validate(observation_dim, n_agents, n_actions)
    qubits = cirq.LineQubit.range(n_agents * d_qubits)
    observables = _joint_observable(qubits)
    quantum_layer = HybridPartiteVariationalEncodingPQC(
        qubits=qubits,
        n_parts=n_agents,
        d_qubits=d_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        trainable_w_enc=trainable_w_enc,
        input_entanglement=True,
        input_entanglement_type=input_entanglement_type,
    )
    return keras.Sequential(
        [
            keras.Input(shape=(n_agents, observation_dim), dtype=tf.float32),
            keras.layers.LocallyConnected1D(
                d_qubits * 3, kernel_size=1, activation=nn_activation
            ),
            keras.layers.Reshape((n_agents, d_qubits, 3)),
            quantum_layer,
            _value_readout(observables, beta),
        ],
        name=name,
    )


def generate_critic_qfctde(
    observation_dim: int,
    n_agents: int = 2,
    n_actions: int = 4,
    d_qubits: int = 4,
    n_layers: int = 5,
    beta: float = 1.0,
    squash_activation: str = "arctan",
    nn_activation: str = "linear",
    trainable_w_enc: bool = False,
    name: str = "critic-qfctde",
) -> keras.Model:
    """Central quantum critic matching the MiniGrid qfCTDE architecture."""
    _validate(observation_dim, n_agents, n_actions)
    total_qubits = n_agents * d_qubits
    qubits = cirq.LineQubit.range(total_qubits)
    observables = _joint_observable(qubits)
    quantum_layer = HybridVariationalEncodingPQC(
        qubits=qubits,
        d_qubits=total_qubits,
        n_layers=n_layers,
        observables=observables,
        squash_activation=squash_activation,
        encoding_layer_cls=ParameterizedRotationLayer_RxRyRz,
        trainable_w_enc=trainable_w_enc,
    )
    return keras.Sequential(
        [
            keras.Input(shape=(n_agents, observation_dim), dtype=tf.float32),
            keras.layers.LocallyConnected1D(
                d_qubits * 3, kernel_size=1, activation=nn_activation
            ),
            keras.layers.Reshape((total_qubits, 3)),
            quantum_layer,
            _value_readout(observables, beta),
        ],
        name=name,
    )


def generate_critic_fctde(
    observation_dim: int,
    n_agents: int = 2,
    n_actions: int = 4,
    units: list[int] | None = None,
    activation: str = "relu",
    name: str = "critic-fctde",
) -> keras.Model:
    """Fully centralized classical critic from the MiniGrid comparison."""
    _validate(observation_dim, n_agents, n_actions)
    units = units or [100]
    layers = [
        keras.Input(shape=(n_agents, observation_dim), dtype=tf.float32),
        keras.layers.Flatten(),
    ]
    layers.extend(keras.layers.Dense(size, activation=activation) for size in units)
    layers.append(keras.layers.Dense(1, name="value"))
    return keras.Sequential(layers, name=name)


def generate_critic_sctde(
    observation_dim: int,
    n_agents: int = 2,
    n_actions: int = 4,
    units: list[int] | None = None,
    activation: str = "relu",
    name: str = "critic-sctde",
) -> keras.Model:
    """Separated classical critic with MiniGrid-style late aggregation."""
    _validate(observation_dim, n_agents, n_actions)
    units = units or [100]
    layers = [keras.Input(shape=(n_agents, observation_dim), dtype=tf.float32)]
    layers.extend(
        keras.layers.LocallyConnected1D(size, kernel_size=1, activation=activation)
        for size in units
    )
    layers.extend([keras.layers.Flatten(), keras.layers.Dense(1, name="value")])
    return keras.Sequential(layers, name=name)
