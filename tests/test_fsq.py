import jax
import jax.numpy as jnp
import numpy as np
import pytest

from popgym_arcade.experiments.memory_models import Branch, fsq, fsq_diagnostics


@pytest.mark.parametrize("levels", [3, 5, 7])
def test_quantizer(levels):
    x = jnp.linspace(-3, 3, 100).astype(jnp.float32)
    for value in (x, x + 1j * x[::-1]):
        q = fsq(value, levels)
        assert q.shape == value.shape and q.dtype == value.dtype
        np.testing.assert_allclose(
            q.real * (levels // 2), jnp.round(q.real * (levels // 2)), atol=1e-6
        )
        if jnp.iscomplexobj(q):
            np.testing.assert_allclose(q.imag, fsq(value.imag, levels))
    grad = jax.grad(lambda z: fsq(z, levels).sum())(x)
    np.testing.assert_allclose(grad, 1 - jnp.tanh(x) ** 2, atol=1e-6)
    assert np.all(np.isfinite(grad)) and np.all(grad > 0)


@pytest.mark.parametrize(
    "mode", ["none", "fsq", "output", "activation", "state_activation"]
)
def test_replay_reset_and_gradient(mode):
    branch = Branch(
        None, "lru", 5, 4, 2, 0.1, jax.random.key(0), recurrent_state_quantization=mode
    )
    x = jax.random.normal(jax.random.key(1), (8, 518))
    done = jnp.array([False, False, True, False, False, True, False, False])
    keys = jax.random.split(jax.random.key(2), 8)
    initial = jnp.ones((2, 4), jnp.complex64)

    def body(s, xs):
        s, y, aux = branch.step(s, *xs, True)
        return s, (y, aux)

    state, (y, _aux) = jax.lax.scan(body, initial, (x, done, keys))
    replay, outputs, _ = branch.parallel_sequence(initial, x, done, keys, True)
    np.testing.assert_allclose(state, replay, atol=2e-6)
    np.testing.assert_allclose(y, outputs, atol=2e-6)
    reset = branch.step(initial, x[0], True, keys[0], True)
    zero = branch.step(jnp.zeros_like(initial), x[0], False, keys[0], True)
    np.testing.assert_allclose(reset[0], zero[0])
    gradient = jax.grad(
        lambda inputs: branch.parallel_sequence(initial, inputs, done, keys, True)[
            1
        ].sum()
    )(x)
    assert np.isfinite(gradient).all() and np.any(gradient != 0)


def test_diagnostics():
    stats = fsq_diagnostics(
        jnp.zeros((4, 3), jnp.complex64), jnp.zeros((4, 3), jnp.complex64)
    )
    assert float(stats["real_error"]) == 0
    assert float(stats["imag_entropy_nats"]) == 0
    np.testing.assert_allclose(stats["real_utilization"], 0.2)


@pytest.mark.parametrize(
    "levels,scale", [(2, 1.0), (1, 1.0), (4, 1.0), (5, 0.0), (5, float("nan"))]
)
def test_invalid_options(levels, scale):
    with pytest.raises(ValueError):
        fsq(jnp.zeros(3), levels, scale)


def test_baseline_formula_and_output_carry():
    from popgym_arcade.experiments.memory_models import LRUCell

    key = jax.random.key(5)
    baseline = LRUCell(3, 4, key)
    output = LRUCell(3, 4, key, "output")
    state = jnp.ones(4, jnp.complex64) * (0.3 + 0.2j)
    x = jnp.array([0.1, 0.4, -0.2])
    expected = (
        jnp.exp(-jnp.exp(baseline.nu) + 1j * jnp.exp(baseline.theta)) * state
        + (jnp.exp(baseline.gamma)[:, None] * (baseline.b_real + 1j * baseline.b_imag))
        @ x
    )
    actual, y = baseline(state, x, jnp.ones(4))
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(actual, output(state, x, jnp.ones(4))[0])
    np.testing.assert_allclose(
        y,
        jax.nn.leaky_relu(
            jnp.real((baseline.c_real + 1j * baseline.c_imag) @ expected)
            + baseline.skip @ x
        ),
    )


def test_dynamics_do_not_requantize_previous_state():
    raw = jnp.full((2, 3), 0.6)
    previous = jnp.full((2, 3), 0.5)
    assert float(fsq_diagnostics(raw, previous)["real_change"]) == 0
    assert (
        float(
            fsq_diagnostics(raw, jnp.zeros_like(raw), valid=jnp.zeros((2, 1)))[
                "real_change"
            ]
        )
        == 0
    )


def test_checkpoint_signature_tracks_fsq_settings():
    from popgym_arcade.experiments.checkpoints import run_signature

    baseline = run_signature({})
    assert baseline == run_signature(
        {"recurrent_state_quantization": "none", "fsq_levels": 5, "fsq_scale": 1.0}
    )
    settings = {
        "recurrent_state_quantization": "fsq",
        "fsq_levels": 5,
        "fsq_scale": 1.0,
    }
    signature = run_signature(settings)
    assert signature != baseline
    for key, value in [
        ("recurrent_state_quantization", "output"),
        ("fsq_levels", 3),
        ("fsq_scale", 2.0),
    ]:
        assert run_signature({**settings, key: value}) != signature


def test_scale_and_known_diagnostic_distribution():
    x = jnp.array([[-2.0, 0.0], [2.0, 0.0]])
    np.testing.assert_array_equal(fsq(x, 5, 2.0), fsq(x / 2.0, 5))
    stats = fsq_diagnostics(x, jnp.zeros_like(x), 3)
    np.testing.assert_allclose(stats["real_utilization"], 0.5)
    np.testing.assert_allclose(stats["real_entropy_nats"], np.log(2) / 2)
    np.testing.assert_allclose(stats["real_saturation"], 0.5)
    np.testing.assert_allclose(stats["real_change"], 0.5)
    np.testing.assert_allclose(stats["real_error"], 1.0)


@pytest.mark.parametrize("mode", ["activation", "state_activation"])
def test_post_activation_placement(mode):
    from popgym_arcade.experiments.memory_models import LRUCell

    key = jax.random.key(42)
    reference = LRUCell(3, 4, key, "fsq" if mode == "state_activation" else "none")
    cell = LRUCell(3, 4, key, mode)
    state = jnp.full(4, 0.2 + 0.4j)
    x = jnp.array([0.5, -0.3, 1.0])
    expected_state, activated = reference(state, x, jnp.ones(4))
    actual_state, output = cell(state, x, jnp.ones(4))
    np.testing.assert_array_equal(actual_state, expected_state)
    np.testing.assert_array_equal(output, fsq(activated))
