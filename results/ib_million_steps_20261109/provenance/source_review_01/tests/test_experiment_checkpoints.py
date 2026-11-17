"""Resume equivalence and publication integrity for periodic PPO checkpoints."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from popgym_arcade.experiments.checkpoints import (
    checkpoint_due,
    restore_checkpoint,
    save_checkpoint,
    trim_training_log,
)


def template():
    network = eqx.nn.Linear(2, 1, key=jax.random.PRNGKey(0))
    tx = optax.adam(optax.linear_schedule(1e-3, 1e-6, 3))
    optimizer = tx.init(eqx.filter(network, eqx.is_array))
    runner = {
        "rng": jax.random.PRNGKey(1),
        "memory": jnp.ones((2,), jnp.complex64),
        "env": jnp.zeros(2),
        "previous_action": jnp.array(0),
        "previous_reward": jnp.array(0.0),
        "done": jnp.array(False),
    }
    return (network, optimizer, runner), tx


def advance(state, tx):
    model, optimizer, runner = state
    rng, key = jax.random.split(runner["rng"])
    x = jax.random.normal(key, (2,))
    loss, grad = eqx.filter_value_and_grad(
        lambda network: jnp.square(network(x)).sum()
    )(model)
    updates, optimizer = tx.update(grad, optimizer, model)
    runner = runner | {
        "rng": rng,
        "memory": runner["memory"] * (0.9 + 0.1j) + x,
        "env": runner["env"] + x,
        "previous_reward": loss,
        "previous_action": jnp.argmax(x),
    }
    return eqx.apply_updates(model, updates), optimizer, runner


def metadata():
    return {
        "variant": "toy",
        "requested_steps": 13,
        "actual_steps": 12,
        "config": {"LR": 1e-3},
    }


def test_resume_matches_uninterrupted_optimizer_rng_and_recurrence(tmp_path):
    initial, tx = template()
    trained = advance(advance(initial, tx), tx)
    path = save_checkpoint(tmp_path, trained, metadata(), 2, 8, 1.0, 5)
    assert (path / "checkpoint.json").exists()
    restored, record = restore_checkpoint(tmp_path, initial, metadata(), 4)
    assert record["completed_updates"] == 2
    expected, actual = advance(trained, tx), advance(restored, tx)
    for left, right in zip(jax.tree.leaves(expected), jax.tree.leaves(actual)):
        np.testing.assert_array_equal(left, right)


def test_corrupted_checkpoint_is_rejected(tmp_path):
    initial, _ = template()
    path = save_checkpoint(tmp_path, initial, metadata(), 1, 4, 1.0, 5)
    with (path / "training_state.eqx").open("ab") as stream:
        stream.write(b"corrupt")
    with pytest.raises(ValueError, match="checksum"):
        restore_checkpoint(tmp_path, initial, metadata(), 4)


def test_configuration_changes_cannot_resume_checkpoint(tmp_path):
    initial, _ = template()
    save_checkpoint(tmp_path, initial, metadata(), 1, 4, 1.0, 5)
    changed = metadata() | {"config": {"LR": 0.1}}
    with pytest.raises(ValueError, match="differs"):
        restore_checkpoint(tmp_path, initial, changed, 4)


def test_partial_checkpoint_and_torn_log_tail(tmp_path):
    initial, _ = template()
    (tmp_path / "checkpoints/.step_000000004.tmp").mkdir(parents=True)
    assert restore_checkpoint(tmp_path, initial, metadata(), 4) is None
    log = tmp_path / "training.jsonl"
    log.write_text('{"update": 1}\n{"update": 2}\n{"update":')
    trim_training_log(log, 2)
    assert log.read_text() == '{"update": 1}\n{"update": 2}\n'
    with pytest.raises(ValueError, match="does not cover"):
        trim_training_log(log, 3)


def test_hundred_thousand_step_cadence():
    steps = [
        update * 2048
        for update in range(1, 489)
        if checkpoint_due((update - 1) * 2048, update * 2048, 100000)
    ]
    assert steps == [
        100352,
        200704,
        301056,
        401408,
        501760,
        600064,
        700416,
        800768,
        901120,
    ]
    assert not checkpoint_due(0, 2048, 0)
    assert 488 * 2048 == 999424


def test_real_environment_runner_and_complex_ib_state_roundtrip(tmp_path):
    import popgym_arcade
    from popgym_arcade.experiments.memory_bias import Runner
    from popgym_arcade.experiments.memory_models import MemoryActorCritic
    from popgym_arcade.wrappers import LogWrapper

    model = MemoryActorCritic(
        "ib_reverse_prior20", jax.random.PRNGKey(5), hidden=6, layers=2
    )
    tx = optax.adam(1e-4)
    optimizer = tx.init(eqx.filter(model, eqx.is_inexact_array))
    env, params = popgym_arcade.make("CartPoleHard", partial_obs=True, obs_size=128)
    env = LogWrapper(env)
    observation, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(6), 1), params
    )
    runner = Runner(
        state,
        observation,
        jnp.zeros(1, jnp.int32),
        jnp.zeros(1),
        jnp.ones(1, bool),
        model.initial_state(1),
        jax.random.PRNGKey(7),
    )
    initial = (model, optimizer, runner)
    save_checkpoint(tmp_path, initial, metadata(), 1, 4, 1.0, 5)
    restored, _ = restore_checkpoint(tmp_path, initial, metadata(), 4)
    for left, right in zip(jax.tree.leaves(initial), jax.tree.leaves(restored)):
        if eqx.is_array(left):
            np.testing.assert_array_equal(left, right)


def test_extension_requires_opt_in_and_only_allows_budget_schedule_changes(tmp_path):
    initial, _ = template()
    old = metadata() | {
        "config": {"LR": 1e-3, "TOTAL_TIMESTEPS": 13},
        "learning_rate_schedule_steps": 13,
    }
    save_checkpoint(tmp_path, initial, old, 2, 8, 1.0, 5)
    extended = old | {
        "requested_steps": 25,
        "actual_steps": 24,
        "config": {"LR": 1e-3, "TOTAL_TIMESTEPS": 25},
        "learning_rate_schedule_steps": 25,
    }
    with pytest.raises(ValueError, match="differs"):
        restore_checkpoint(tmp_path, initial, extended, 4)
    restored, record = restore_checkpoint(
        tmp_path, initial, extended, 4, allow_budget_extension=True
    )
    assert record["completed_updates"] == 2
    for left, right in zip(jax.tree.leaves(initial), jax.tree.leaves(restored)):
        if eqx.is_array(left):
            np.testing.assert_array_equal(left, right)
    incompatible = extended | {"config": {"LR": 0.1, "TOTAL_TIMESTEPS": 25}}
    with pytest.raises(ValueError, match="differs"):
        restore_checkpoint(
            tmp_path, initial, incompatible, 4, allow_budget_extension=True
        )


def test_gradient_metrics_use_complex_magnitude_and_clipping():
    from popgym_arcade.experiments.memory_bias import gradient_norm_metrics

    class Gradients(eqx.Module):
        actor: jax.Array
        critic: jax.Array

    gradients = Gradients(jnp.array([3 + 4j]), jnp.array([12.0]))
    np.testing.assert_allclose(
        gradient_norm_metrics(gradients, 0.5), [13, 13, 0.5, 5, 12]
    )
