"""Action selection, reset boundaries, replay, and auxiliary gradient isolation."""

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from popgym_arcade.experiments.memory_models import MemoryActorCritic, prediction_losses


def test_selected_action_and_terminal_targets():
    p = jnp.zeros((2, 5, 513))
    action = jnp.array([1, 3])
    embedding = jnp.ones((2, 512)).at[1].set(999)
    done = jnp.array([False, True])
    reward = jnp.array([2.0, 4.0])
    ol, rl = prediction_losses(p, action, embedding, reward, done)
    np.testing.assert_allclose([ol, rl], [1, 10])
    grads = jax.grad(
        lambda x: sum(prediction_losses(x, action, embedding, reward, done))
    )(p)
    assert jnp.all(grads[:, 0] == 0)
    assert jnp.all(grads[1, :, :512] == 0)
    assert grads[1, 3, 512] != 0
    assert prediction_losses(p, action, embedding, reward, jnp.ones(2, bool))[0] == 0


def test_prediction_replay_and_encoder_gradients():
    model = MemoryActorCritic(
        "lru_prediction", jax.random.PRNGKey(0), hidden=6, layers=2
    )
    cnn = eqx.nn.Sequential(
        [eqx.nn.Lambda(jnp.ravel), eqx.nn.Linear(192, 512, key=jax.random.PRNGKey(1))]
    )
    model = eqx.tree_at(lambda m: (m.actor.cnn, m.critic.cnn), model, (cnn, cnn))
    obs = jax.random.randint(jax.random.PRNGKey(2), (3, 1, 8, 8, 3), 0, 256)
    action = jnp.zeros((3, 1), jnp.int32)
    reward = jnp.zeros((3, 1))
    done = jnp.array([[True], [False], [True]])
    keys = jax.random.split(jax.random.PRNGKey(3), (3, 1))
    initial = model.initial_state(1)
    _, (_, _, parallel) = model.sequence(initial, obs, action, reward, done, keys)

    def step(state, xs):
        state, _, _, aux = model.step(state, *xs, True)
        return state, aux

    _, serial = jax.lax.scan(step, initial, (obs, action, reward, done, keys))
    for a, b in zip(parallel, serial):
        np.testing.assert_allclose(a, b, rtol=2e-5, atol=2e-5)

    def objective(m, detach):
        _, (_, _, predictions) = m.sequence(
            initial, obs, action, reward, done, keys, detach_encoder=detach
        )
        targets = m.encode(
            obs.reshape(3, 8, 8, 3), action.ravel(), reward.ravel(), done.ravel()
        )
        return sum(
            prediction_losses(p, action, x[:, -512:].reshape(3, 1, 512), reward, done)[
                0
            ]
            for p, x in zip(predictions, targets)
        )

    grads = eqx.filter_grad(lambda m: objective(m, True))(model)
    norm = lambda x: sum(
        float(jnp.sum(jnp.abs(a))) for a in jax.tree.leaves(eqx.filter(x, eqx.is_array))
    )
    assert norm(grads.actor.cnn) == norm(grads.critic.cnn) == 0
    assert norm(grads.actor.cells) > 0
    assert norm(grads.actor.prediction) > 0
    assert norm(eqx.filter_grad(lambda m: objective(m, False))(model).actor.cnn) > 0
