"""Manual CPU smoke: real environment rollout and one PPO update per mode."""

import time

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import popgym_arcade
from popgym_arcade.experiments.memory_bias import Runner, build_update, upstream_config
from popgym_arcade.experiments.memory_models import MemoryActorCritic
from popgym_arcade.wrappers import LogWrapper


def main():
    config = upstream_config()
    config.update(NUM_ENVS=1, NUM_STEPS=2, NUM_MINIBATCHES=1, UPDATE_EPOCHS=1)
    env, params = popgym_arcade.make("CartPoleHard", partial_obs=True, obs_size=128)
    env = LogWrapper(env)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(jax.random.PRNGKey(1), 1), params
    )
    for mode in ("none", "fsq", "output"):
        model = MemoryActorCritic(
            "lru",
            jax.random.PRNGKey(0),
            hidden=4,
            layers=1,
            recurrent_state_quantization=mode,
        )
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        opt = tx.init(eqx.filter(model, eqx.is_array))
        runner = Runner(
            state,
            obs,
            jnp.zeros(1, jnp.int32),
            jnp.zeros(1),
            jnp.ones(1, bool),
            model.initial_state(1),
            jax.random.PRNGKey(2),
        )
        update = build_update(env, params, config, tx, 1.0)
        start = time.monotonic()
        model, opt, runner, metrics, _, _ = update(model, opt, runner)
        metrics = np.asarray(metrics)
        assert np.isfinite(metrics).all()
        assert len(metrics) == (9 if mode == "none" else 29)
        assert all(
            np.isfinite(x).all()
            for x in jax.tree.leaves(eqx.filter(model, eqx.is_array))
        )
        print(
            mode,
            "finite PPO update; seconds including compilation:",
            time.monotonic() - start,
            flush=True,
        )


if __name__ == "__main__":
    main()
