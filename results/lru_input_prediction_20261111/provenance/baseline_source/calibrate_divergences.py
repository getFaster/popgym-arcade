"""Frozen RMS calibration on one shared random-policy trajectory per setting."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

import popgym_arcade
from popgym_arcade.experiments.memory_bias import upstream_config
from popgym_arcade.experiments.memory_models import MemoryActorCritic
from popgym_arcade.wrappers import LogWrapper

KINDS = {
    "forward_kl": "information_bottleneck",
    "reverse_kl": "ib_reverse_kl",
    "symmetric_kl": "ib_symmetric_kl",
    "renyi_half": "ib_renyi_half",
}


def calibrate(
    env_name,
    partial_obs,
    posterior_input,
    seed,
    output,
    *,
    hidden=512,
    layers=2,
    residual=True,
    config=None,
):
    config = upstream_config() if config is None else config
    n, steps = config["NUM_ENVS"], config["NUM_STEPS"]
    env, params = popgym_arcade.make(
        env_name, partial_obs=partial_obs, obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    rng = jax.random.PRNGKey(seed + 900_000)
    rng, key = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(key, n), params
    )

    @eqx.filter_jit
    def collect(state, obs, rng):
        def body(carry, _):
            state, obs, action, reward, done, rng = carry
            rng, ak, ek, mk = jax.random.split(rng, 4)
            row = (obs, action, reward, done, jax.random.split(mk, n))
            action = jax.random.randint(ak, (n,), 0, 5)
            obs, state, reward, done, _ = jax.vmap(env.step, in_axes=(0, 0, 0, None))(
                jax.random.split(ek, n), state, action, params
            )
            return (state, obs, action, reward, done, rng), row

        return jax.lax.scan(
            body,
            (state, obs, jnp.zeros(n, jnp.int32), jnp.zeros(n), jnp.ones(n, bool), rng),
            None,
            length=steps,
        )[1]

    batch = collect(state, obs, rng)
    batch_hash = hashlib.sha256()
    for value in batch:
        batch_hash.update(np.asarray(value).tobytes())
    network = MemoryActorCritic(
        "information_bottleneck",
        jax.random.PRNGKey(seed),
        posterior_input=posterior_input,
        residual=residual,
        hidden=hidden,
        layers=layers,
        obs_size=config["OBS_SIZE"],
    )

    @eqx.filter_jit
    def encode_batch(model, data):
        return jax.lax.map(lambda row: model.encode(*row), data[:4])

    encoded = encode_batch(network, batch)
    jax.clear_caches()
    scales, raw = {}, {}
    for kind, variant in KINDS.items():
        branches = []
        for branch in (network.actor, network.critic):
            branch = copy.copy(branch)
            object.__setattr__(branch, "variant", variant)
            branches.append(branch)
        model = eqx.tree_at(lambda m: (m.actor, m.critic), network, tuple(branches))

        @eqx.filter_jit
        def replay(m, inputs, done, keys):
            values = []
            for branch, x, initial in zip(
                (m.actor, m.critic), inputs, m.initial_state(n)
            ):
                result = jax.vmap(
                    lambda state, sequence, resets, rngs, branch=branch: (
                        branch.parallel_sequence(state, sequence, resets, rngs, False)[
                            2
                        ]
                    ),
                    in_axes=(0, 1, 1, 1),
                    out_axes=1,
                )(initial, x, done, keys)
                values.append(result)
            return jnp.stack(values)

        values = np.asarray(
            replay(model, encoded, batch[3], batch[4]), dtype=np.float64
        ) / np.log(2.0)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite calibration divergence")
        scales[kind] = np.maximum(
            np.sqrt(np.mean(values**2, axis=(1, 2))), 1e-8
        ).tolist()
        raw[kind] = {
            "mean_bits": values.mean(axis=(1, 2)).tolist(),
            "max_bits": values.max(axis=(1, 2)).tolist(),
        }
        jax.clear_caches()
    result = {
        "env": env_name,
        "partial_obs": partial_obs,
        "posterior_input": posterior_input,
        "seed": seed,
        "residual": residual,
        "hidden": hidden,
        "layers": layers,
        "obs_size": config["OBS_SIZE"],
        "samples": n * steps,
        "policy": "uniform_random_5_actions",
        "calibration_seed": seed + 900_000,
        "batch_sha256": batch_hash.hexdigest(),
        "scales_bits": scales,
        "raw": raw,
        "minimum_scale_bits": 1e-8,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True)
    parser.add_argument("--partial-obs", action="store_true")
    parser.add_argument("--posterior-input", choices=["state", "output"], required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    calibrate(args.env, args.partial_obs, args.posterior_input, args.seed, args.output)


if __name__ == "__main__":
    main()
