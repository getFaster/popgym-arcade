"""Scientific invariants for the memory-bias controls (CPU compatible)."""

import os

os.environ["JAX_PLATFORMS"] = "cpu"

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from popgym_arcade.experiments.memory_models import (
    IB_DIVERGENCES,
    Branch,
    diagonal_kl,
    free_bit,
    gaussian_divergence,
    make_input,
)


def test_input_order_and_reset():
    x = make_input(
        jnp.array([[10.0, 20.0], [30.0, 40.0]]),
        jnp.array([3, 2]),
        jnp.array([0.7, 0.8]),
        jnp.array([False, True]),
    )
    np.testing.assert_allclose(x[0], [0, 0, 0, 1, 0, 0.7, 10, 20])
    np.testing.assert_allclose(x[1], [0, 0, 0, 0, 0, 0, 30, 40])


def test_kl_direction_and_free_bit():
    zero = jnp.zeros(3)
    log_four = jnp.full(3, jnp.log(4.0))
    forward = diagonal_kl(zero, zero, zero, log_four)
    reverse = diagonal_kl(zero, log_four, zero, zero)
    np.testing.assert_allclose(forward, 1.5 * (np.log(4) + 0.25 - 1), rtol=1e-6)
    assert not np.isclose(forward, reverse)
    assert float(free_bit(0.0)) == 1
    assert float(free_bit(jnp.log(2.0))) == 1
    np.testing.assert_allclose(free_bit(2 * jnp.log(2.0)), 2.0)
    assert float(jax.grad(lambda x: free_bit(x))(0.1)) == 0


def make_branch(variant):
    return Branch(eqx.nn.Sequential([]), variant, 5, 6, 2, 0.5, jax.random.PRNGKey(0))


def test_ib_parameters_and_state_independent_heads():
    branch = make_branch("information_bottleneck")
    assert branch.latent_dim == (2 * (2 * 6 * 2)) // 3
    assert branch.head.layers[0].in_features == 518 + 2 * branch.latent_dim
    x = jax.random.normal(jax.random.PRNGKey(1), (518,))
    state = jnp.ones((2, 6), jnp.complex64)
    key = jax.random.PRNGKey(2)
    _, logits0, kl0 = branch.step(state * 0, x, False, key, True)
    _, logits1, kl1 = branch.step(state, x, False, key, True)
    np.testing.assert_array_equal(logits0, logits1)
    assert not np.isclose(kl0, kl1)
    assert jnp.isfinite(kl0)


@pytest.mark.parametrize("variant", ["lru", "state_dropout", *IB_DIVERGENCES])
def test_reset_and_replay(variant):
    branch = make_branch(variant)
    x = jnp.ones(518)
    state = jnp.ones((2, 6), jnp.complex64)
    key = jax.random.PRNGKey(4)
    reset = branch.step(state, x, True, key, True)
    fresh = branch.step(jnp.zeros_like(state), x, False, key, True)
    replay = branch.step(state, x, True, key, True)
    for actual, expected, repeated in zip(reset, fresh, replay):
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(actual, repeated)


def test_dropout_changes_transition_and_eval_disables_it():
    branch = make_branch("state_dropout")
    state, x = jnp.ones((2, 6), jnp.complex64), jnp.ones(518)
    k1, k2 = jax.random.split(jax.random.PRNGKey(5))
    assert not np.array_equal(
        branch.step(state, x, False, k1, True)[0],
        branch.step(state, x, False, k2, True)[0],
    )
    np.testing.assert_array_equal(
        branch.step(state, x, False, k1, False)[0],
        branch.step(state, x, False, k2, False)[0],
    )


def test_gae_stops_at_episode_boundary():
    from popgym_arcade.experiments.memory_bias import (
        Transition,
        advantages_and_targets,
        upstream_config,
    )

    dummy = jnp.zeros((2, 1))
    batch = Transition(
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        dummy,
        jnp.array([[1.0], [2.0]]),
        jnp.array([[1.0], [3.0]]),
        jnp.array([[False], [True]]),
        dummy,
        dummy,
    )
    advantage, target = advantages_and_targets(batch, jnp.array([999.0]), 0.99, 0.95)
    np.testing.assert_allclose(advantage[:, 0], [2.9205, 1.0], rtol=1e-6)
    np.testing.assert_allclose(target[:, 0], [3.9205, 3.0], rtol=1e-6)
    config = upstream_config()
    assert config["NUM_ENVS"] == 16
    assert config["NUM_STEPS"] == 128
    assert config["TOTAL_TIMESTEPS"] == 10_000_000


@pytest.mark.parametrize("variant", ["lru", "state_dropout", *IB_DIVERGENCES])
def test_rollout_probabilities_match_sequence_replay(variant):
    from popgym_arcade.experiments.memory_models import MemoryActorCritic

    model = MemoryActorCritic(variant, jax.random.PRNGKey(9), hidden=6, layers=2)
    obs = jax.random.randint(
        jax.random.PRNGKey(10), (3, 2, 128, 128, 3), 0, 256
    ).astype(jnp.uint8)
    action, reward = jnp.ones((3, 2), jnp.int32), jnp.ones((3, 2)) * 0.1
    done = jnp.array([[True, True], [False, False], [False, True]])
    keys = jax.random.split(jax.random.PRNGKey(11), (3, 2))
    initial = model.initial_state(2)

    @eqx.filter_jit
    def compare(model):
        def body(state, inputs):
            o, a, r, d, k = inputs
            state, logits, value, _ = model.step(state, o, a, r, d, k, True)
            return state, (logits, value)

        final, outputs = jax.lax.scan(body, initial, (obs, action, reward, done, keys))
        replay_final, (replay_logits, replay_value, _) = model.sequence(
            initial, obs, action, reward, done, keys
        )
        return final, outputs, replay_final, (replay_logits, replay_value)

    final, outputs, replay_final, replay_outputs = compare(model)
    for actual, expected in zip(
        jax.tree.leaves((final, outputs)),
        jax.tree.leaves((replay_final, replay_outputs)),
    ):
        np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)


def test_complex_lru_gradients_match_finite_differences():
    from popgym_arcade.experiments.memory_models import LRUCell

    cell = LRUCell(4, 3, jax.random.PRNGKey(17))
    state = jnp.array([0.1 + 0.2j, -0.2 + 0.1j, 0.3 - 0.1j])
    x = jnp.array([0.2, -0.3, 0.4, 0.1])

    def loss(model):
        _, output = model(state, x, jnp.ones(3))
        return jnp.sum(output**2)

    grad = eqx.filter_grad(loss)(cell)
    for name, index in [
        ("nu", (0,)),
        ("theta", (0,)),
        ("b_real", (0, 0)),
        ("b_imag", (0, 0)),
    ]:
        parameter = getattr(cell, name)
        eps = 0.001
        plus = eqx.tree_at(
            lambda m, name=name: getattr(m, name), cell, parameter.at[index].add(eps)
        )
        minus = eqx.tree_at(
            lambda m, name=name: getattr(m, name), cell, parameter.at[index].add(-eps)
        )
        numerical = (loss(plus) - loss(minus)) / (2 * eps)
        np.testing.assert_allclose(
            getattr(grad, name)[index], numerical, rtol=0.02, atol=2e-5
        )


@pytest.mark.parametrize(
    "kind", ["forward_kl", "reverse_kl", "symmetric_kl", "renyi_half"]
)
def test_divergence_identity_symmetry_and_gradients(kind):
    pm, pl = jnp.array([0.2, -0.5]), jnp.array([0.3, -0.4])
    qm, ql = jnp.array([-0.3, 0.7]), jnp.array([-0.2, 0.8])
    divergence = gaussian_divergence(pm, pl, qm, ql, kind)
    assert float(divergence) >= 0
    np.testing.assert_allclose(gaussian_divergence(pm, pl, pm, pl, kind), 0, atol=1e-6)
    if kind in ("renyi_half", "symmetric_kl"):
        np.testing.assert_allclose(
            divergence, gaussian_divergence(qm, ql, pm, pl, kind), rtol=1e-6
        )
    gradient = jax.grad(lambda mean: gaussian_divergence(mean, pl, qm, ql, kind))(pm)
    assert np.isfinite(gradient).all()
    assert np.linalg.norm(gradient) > 0


def test_renyi_half_matches_independent_gaussian_overlap_integral():
    # Numerical quadrature of sqrt(p q), independent of the closed-form formula.
    from scipy.integrate import quad
    from scipy.stats import norm

    pm, pv, qm, qv = 0.4, 0.7, -1.2, 2.3
    overlap, _ = quad(
        lambda x: np.sqrt(norm.pdf(x, pm, np.sqrt(pv)) * norm.pdf(x, qm, np.sqrt(qv))),
        -np.inf,
        np.inf,
    )
    expected = -2 * np.log(overlap)
    actual = gaussian_divergence(
        jnp.array([pm]),
        jnp.log(jnp.array([pv])),
        jnp.array([qm]),
        jnp.log(jnp.array([qv])),
        "renyi_half",
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-6)
    assert actual <= gaussian_divergence(
        jnp.array([pm]),
        jnp.log(jnp.array([pv])),
        jnp.array([qm]),
        jnp.log(jnp.array([qv])),
        "forward_kl",
    )


def test_task_specific_evaluation_horizons():
    import popgym_arcade
    from popgym_arcade.experiments.memory_bias import episode_horizon

    for name, horizon in [
        ("CartPoleHard", 600),
        ("NavigatorEasy", 64),
        ("MineSweeperEasy", 32),
        ("BreakoutEasy", 2000),
    ]:
        env, params = popgym_arcade.make(name, partial_obs=False, obs_size=128)
        assert episode_horizon(env, params) == horizon


def test_ib_divergences_share_initial_parameters():
    branches = [make_branch(variant) for variant in IB_DIVERGENCES]
    reference = jax.tree.leaves(eqx.filter(branches[0], eqx.is_array))
    for branch in branches[1:]:
        leaves = jax.tree.leaves(eqx.filter(branch, eqx.is_array))
        assert len(leaves) == len(reference)
        for actual, expected in zip(leaves, reference):
            np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize(
    "kind", ["forward_kl", "reverse_kl", "symmetric_kl", "renyi_half"]
)
def test_every_divergence_has_free_region_and_nonzero_active_gradient(kind):
    zeros = jnp.zeros(3)

    def loss(mean):
        return free_bit(gaussian_divergence(mean, zeros, zeros, zeros, kind))

    assert float(loss(jnp.ones(3) * 0.01)) == 1
    np.testing.assert_array_equal(jax.grad(loss)(jnp.ones(3) * 0.01), zeros)
    assert float(loss(jnp.ones(3) * 3)) > 1
    assert np.linalg.norm(jax.grad(loss)(jnp.ones(3) * 3)) > 0


def test_evaluation_allows_terminal_step_after_nominal_horizon():
    from popgym_arcade.experiments.memory_bias import evaluate

    class LateTerminalEnv:
        def reset(self, key, params):
            return jnp.zeros(1), jnp.array(0, jnp.int32)

        def step(self, key, state, action, params):
            state = state + 1
            done = state == 33
            info = {"returned_episode_returns": jnp.where(done, 33.0, 0.0)}
            return jnp.zeros(1), jnp.where(done, 0, state), jnp.array(1.0), done, info

    class Policy(eqx.Module):
        def initial_state(self, n):
            return jnp.zeros(n)

        def step(self, state, obs, action, reward, done, keys, training):
            n = obs.shape[0]
            return state, jnp.zeros((n, 5)), jnp.zeros(n), (jnp.zeros(n), jnp.zeros(n))

    scores = evaluate(Policy(), LateTerminalEnv(), None, 0, 16, 4, 32)
    np.testing.assert_array_equal(scores, jnp.full(64, 33.0))
