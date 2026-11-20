import jax
import jax.numpy as jnp
import numpy as np
import pytest
from distreqx import distributions

from plotting.density_analysis_memfix import bin_distribution
from plotting.density_analysis_ppo import normalize_recall_gradients
from popgym_arcade.experiments.memory_models import Branch


def test_upstream_normalization_and_zero_convention():
    grads = jnp.array([[[1.0, -2.0]], [[-3.0, 4.0]]])
    np.testing.assert_allclose(normalize_recall_gradients(grads), [0.3, 0.7])
    np.testing.assert_array_equal(normalize_recall_gradients(grads * 0), [0, 0])


def test_sixteen_bins_include_endpoint_and_conserve_mass():
    mass = np.arange(1, 18, dtype=float)
    mass /= mass.sum()
    binned = bin_distribution(mass)
    assert binned.shape == (16,)
    np.testing.assert_allclose(binned.sum(), 1)
    np.testing.assert_allclose(binned[-1], mass[-2:].sum())
    np.testing.assert_array_equal(bin_distribution([1]), [1] + [0] * 15)


@pytest.mark.parametrize("mode", ["none", "activation", "fsq"])
def test_recall_sequential_parallel_and_reset(mode):
    branch = Branch(
        None, "lru", 5, 4, 2, 0.1, jax.random.key(0), recurrent_state_quantization=mode
    )
    x = jax.random.normal(jax.random.key(1), (8, 518))
    done = jnp.array([True, False, False, True, False, False, False, False])
    keys = jax.random.split(jax.random.key(2), 8)
    initial = jnp.zeros((2, 4), jnp.complex64)

    def target(inputs, parallel):
        if parallel:
            _, logits, _ = branch.parallel_sequence(initial, inputs, done, keys, False)
        else:

            def body(s, xs):
                s, logits, _ = branch.step(s, *xs, False)
                return s, logits

            _, logits = jax.lax.scan(body, initial, (inputs, done, keys))
        return distributions.Categorical(logits=logits[-1]).logits.sum()

    sequential = jax.grad(lambda inputs: target(inputs, False))(x)
    parallel = jax.grad(lambda inputs: target(inputs, True))(x)
    np.testing.assert_allclose(sequential, parallel, atol=3e-6)
    np.testing.assert_array_equal(sequential[:3], 0)
    assert np.any(np.asarray(sequential[3:]) != 0)


def test_paper_actionwise_norm_prevents_probability_gradient_cancellation():
    def probabilities(x):
        return jax.nn.softmax(jnp.array([x[0], -x[0]]))

    images = jnp.array([0.3])
    jacobian = jax.jacrev(probabilities)(images)
    assert float(jnp.abs(jacobian).sum()) > 0
    np.testing.assert_allclose(
        jax.grad(lambda x: probabilities(x).sum())(images), 0, atol=1e-7
    )


def test_shared_endpoint_plan():
    from plotting.recall_queue import make_plan

    first = make_plan()
    assert first == make_plan()
    assert len(first["endpoints"]) == len(set(first["endpoints"])) == 8
    assert all(9 <= t < first["horizon"] for t in first["endpoints"])
    assert len(first["trajectory_seeds"]) == 8


def test_sampled_endpoint_and_sequential_action_gradients():
    import equinox as eqx

    from plotting.density_analysis_memfix import recall

    class ToyActor(eqx.Module):
        layers: int = 1
        hidden: int = 1

        def encode(self, obs, action, reward, done):
            return obs.reshape(len(obs), -1)

        def step(self, memory, x, done, key, training):
            memory = jnp.where(done, 0, memory) + x[0]
            return memory, memory.real[0, 0] * jnp.arange(5), None

    obs = jnp.arange(10, dtype=float).reshape(10, 1, 1, 1) * 0.03
    done = jnp.arange(10) == 0
    endpoint = 3
    actual = recall(
        ToyActor(),
        obs,
        jnp.zeros(10, int),
        jnp.zeros(10),
        done,
        definition="paper",
        endpoint=jnp.array(endpoint),
    )

    def target(images):
        return jax.nn.softmax(images[: endpoint + 1].sum() * jnp.arange(5))

    gradients = jnp.moveaxis(jax.jacrev(target)(obs), 1, 0)
    expected = normalize_recall_gradients(gradients)
    np.testing.assert_allclose(actual, expected, atol=1e-7)
    np.testing.assert_array_equal(actual[endpoint + 1 :], 0)


def test_sliding_window_preserves_rollout_carry(monkeypatch):
    import equinox as eqx

    from plotting import density_analysis_memfix as diagnostic

    class ToyActor(eqx.Module):
        layers: int = 1
        hidden: int = 1

        def encode(self, obs, action, reward, done):
            return obs.reshape(len(obs), -1)

        def step(self, memory, x, done, key, training):
            memory = jnp.where(done, 0, memory) + x[0]
            return memory, memory.real[0, 0] * jnp.arange(5), None

    class Model:
        actor = ToyActor()

        def initial_state(self, batch):
            return (jnp.zeros((batch, 1, 1), jnp.complex64),) * 2

    class Env:
        def reset(self, key, params):
            return jnp.ones((1, 1, 1)), 0

        def step(self, key, state, action, params):
            obs = jnp.full((1, 1, 1), float(state + 2))
            return obs, state + 1, jnp.array(0.0), False, {}

    monkeypatch.setattr(diagnostic, "step_env", lambda env, *args: env.step(*args))
    history, carry, collected = diagnostic.collect(
        Model(), Env(), None, 0, 12, window=10
    )
    assert collected == 12 and len(history[0]) == 10
    np.testing.assert_array_equal(history[0].ravel(), np.arange(3, 13))
    np.testing.assert_allclose(carry, [[3]])
    assert not np.asarray(history[-1]).any()
    # The sampled window's forward pass reproduces the actual rollout state.
    state = carry
    for inputs in zip(*history):
        state, _ = diagnostic.actor_step(Model.actor, state, *inputs)
    np.testing.assert_allclose(state, [[78]])


@pytest.mark.parametrize(
    "requested,collected,fallback,expected,status",
    [
        (23, 24, True, 23, "defined"),
        (23, 17, True, 16, "defined_terminal_fallback"),
        (23, 10, True, 9, "defined_terminal_fallback"),
        (23, 9, True, None, "insufficient_history"),
        (23, 17, False, None, "terminated_before_endpoint"),
    ],
)
def test_last_valid_endpoint_fallback(requested, collected, fallback, expected, status):
    from plotting.density_analysis_memfix import choose_endpoint

    assert choose_endpoint(requested, collected, 10, fallback) == (expected, status)


def test_plain_lru_is_separate_from_quantized_lru():
    from plotting.density_analysis_memfix import family
    from plotting.report_main_recall import matches_variant

    plain = {"variant": "lru", "quantization": "none"}
    quantized = {"variant": "lru", "quantization": "activation", "levels": 5}
    assert matches_variant(plain, "lru")
    assert not matches_variant(quantized, "lru")
    assert matches_variant(quantized, "activation_5")
    assert not matches_variant(plain, "activation_5")
    assert family({"variant": "lru"}) == "Plain LRU"
    assert (
        family({"variant": "lru", "recurrent_state_quantization": "activation"})
        == "FSQ"
    )
