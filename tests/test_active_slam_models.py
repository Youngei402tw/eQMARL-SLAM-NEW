import pytest

from eqmarl.active_slam_models import (
    generate_actor_classical as generate_faithful_actor,
    generate_critic_eqmarl as generate_faithful_eqmarl_critic,
    generate_critic_fctde as generate_faithful_fctde_critic,
    generate_critic_qfctde as generate_faithful_qfctde_critic,
    generate_critic_sctde as generate_faithful_sctde_critic,
)

from eqmarl.models import (
    generate_model_active_slam_actor_classical,
    generate_model_active_slam_actor_quantum,
    generate_model_active_slam_critic_classical,
    generate_model_active_slam_critic_fctde,
    generate_model_active_slam_critic_quantum_partite,
    generate_model_active_slam_critic_sctde,
)


def test_classical_model_shapes():
    actor = generate_model_active_slam_actor_classical(411)
    critic = generate_model_active_slam_critic_fctde(411)
    separated_critic = generate_model_active_slam_critic_sctde(411)
    assert actor.compute_output_shape((None, 411)).as_list() == [None, 4]
    assert critic.compute_output_shape((None, 2, 411)).as_list() == [None, 1]
    assert separated_critic.compute_output_shape((None, 2, 411)).as_list() == [None, 1]


def test_classical_critic_architectures_are_distinct_and_parameter_matched():
    fully_centralized = generate_model_active_slam_critic_fctde(411)
    separated = generate_model_active_slam_critic_sctde(411)
    legacy = generate_model_active_slam_critic_classical(411)
    assert fully_centralized.layers[0].__class__.__name__ == "Flatten"
    assert separated.layers[0].__class__.__name__ == "LocallyConnected1D"
    assert legacy.count_params() == fully_centralized.count_params()
    assert abs(separated.count_params() - fully_centralized.count_params()) < 100


def test_action_observable_validation():
    with pytest.raises(ValueError, match="at least n_actions"):
        generate_model_active_slam_actor_quantum(411, n_actions=5, d_qubits=4)


def test_quantum_model_shapes():
    actor = generate_model_active_slam_actor_quantum(411, n_layers=1)
    critic = generate_model_active_slam_critic_quantum_partite(411, n_layers=1)
    assert actor.compute_output_shape((None, 411)).as_list() == [None, 4]
    assert critic.compute_output_shape((None, 2, 411)).as_list() == [None, 1]


def test_minigrid_faithful_active_slam_shapes_and_parameter_budgets():
    actor = generate_faithful_actor(147)
    eqmarl = generate_faithful_eqmarl_critic(147, n_layers=1)
    qfctde = generate_faithful_qfctde_critic(147, n_layers=1)
    fctde = generate_faithful_fctde_critic(147)
    sctde = generate_faithful_sctde_critic(147)
    assert actor.compute_output_shape((None, 147)).as_list() == [None, 3]
    assert eqmarl.compute_output_shape((None, 2, 147)).as_list() == [None, 1]
    assert qfctde.count_params() == eqmarl.count_params()
    assert len(eqmarl.trainable_variables) == len(qfctde.trainable_variables) == 4
    assert fctde.count_params() == 29601
    assert sctde.count_params() == 29801
