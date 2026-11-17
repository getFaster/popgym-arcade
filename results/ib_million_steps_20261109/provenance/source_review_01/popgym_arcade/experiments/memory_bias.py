"""CUDA PPO experiment: MLP reference and four LRU conditions.

Run with uv run python -m popgym_arcade.experiments.memory_bias --help.
"""

import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import NamedTuple

# Require GPU; never silently run a comparison on CPU.
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import popgym_arcade
from popgym_arcade.experiments.checkpoints import (
    checkpoint_due,
    restore_checkpoint,
    save_checkpoint,
    trim_training_log,
)
from popgym_arcade.experiments.memory_models import (
    DEFAULT_IB_NORMALIZATION,
    DEFAULT_IB_VARIANT,
    DEFAULT_VARIANTS,
    IB_DIVERGENCES,
    KL_BALANCES,
    VARIANTS,
    MemoryActorCritic,
    free_bit,
    normalized_excess,
    resolve_memory_options,
)
from popgym_arcade.train import get_args
from popgym_arcade.wrappers import LogWrapper


class Transition(NamedTuple):
    obs: jax.Array
    previous_action: jax.Array
    previous_reward: jax.Array
    done: jax.Array
    keys: jax.Array
    action: jax.Array
    value: jax.Array
    reward: jax.Array
    next_done: jax.Array
    log_prob: jax.Array
    episode_return: jax.Array


class Runner(NamedTuple):
    env_state: object
    obs: jax.Array
    previous_action: jax.Array
    previous_reward: jax.Array
    done: jax.Array
    memory: tuple
    rng: jax.Array


def upstream_config():
    """Read the actual parser defaults, so settings cannot silently drift."""
    argv = sys.argv
    try:
        sys.argv = [argv[0], "PPO_RNN"]
        return vars(get_args())
    finally:
        sys.argv = argv


def log_prob(logits, action):
    return jnp.take_along_axis(jax.nn.log_softmax(logits), action[..., None], -1)[
        ..., 0
    ]


def advantages_and_targets(batch, last_value, gamma, gae_lambda):
    def body(carry, transition):
        gae, next_value = carry
        alive = 1 - transition.next_done
        delta = transition.reward + gamma * next_value * alive - transition.value
        gae = delta + gamma * gae_lambda * alive * gae
        return (gae, transition.value), gae

    _, advantages = jax.lax.scan(
        body, (jnp.zeros_like(last_value), last_value), batch, reverse=True
    )
    return advantages, advantages + batch.value


def gradient_norm_metrics(grads, max_norm):
    raw = optax.global_norm(grads)
    return jnp.stack(
        (
            raw,
            raw,
            jnp.minimum(raw, max_norm),
            optax.global_norm(grads.actor),
            optax.global_norm(grads.critic),
        )
    )


def build_update(
    env, params, config, tx, ib_coefficient, rms_scales=None, log_gradients=False
):
    n, steps = config["NUM_ENVS"], config["NUM_STEPS"]
    minibatches = config["NUM_MINIBATCHES"]
    assert n % minibatches == 0, "Recurrent minibatches must contain whole sequences"

    @eqx.filter_jit
    def update(network, opt_state, runner):
        initial_memory = runner.memory

        def rollout(r, _):
            rng, model_key, action_key, env_key = jax.random.split(r.rng, 4)
            keys = jax.random.split(model_key, n)
            memory, logits, value, _ = network.step(
                r.memory,
                r.obs,
                r.previous_action,
                r.previous_reward,
                r.done,
                keys,
                True,
            )
            action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            obs, state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(env_key, n), r.env_state, action, params)
            transition = Transition(
                r.obs,
                r.previous_action,
                r.previous_reward,
                r.done,
                keys,
                action,
                value,
                reward,
                done,
                log_prob(logits, action),
                info["returned_episode_returns"],
            )
            return Runner(state, obs, action, reward, done, memory, rng), transition

        runner, batch = jax.lax.scan(rollout, runner, None, length=steps)
        _, _, last_value, _ = network.step(
            runner.memory,
            runner.obs,
            runner.previous_action,
            runner.previous_reward,
            runner.done,
            jax.random.split(runner.rng, n),
            True,
        )
        advantages, targets = advantages_and_targets(
            batch, last_value, config["GAMMA"], config["GAE_LAMBDA"]
        )

        def loss(model, memory, trajectory, advantage, target):
            _, (logits, value, kl) = model.sequence(
                memory,
                trajectory.obs,
                trajectory.previous_action,
                trajectory.previous_reward,
                trajectory.done,
                trajectory.keys,
            )
            ratio = jnp.exp(log_prob(logits, trajectory.action) - trajectory.log_prob)
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
            policy_loss = -jnp.minimum(
                ratio * advantage,
                jnp.clip(ratio, 1 - config["CLIP_EPS"], 1 + config["CLIP_EPS"])
                * advantage,
            ).mean()
            clipped_value = trajectory.value + jnp.clip(
                value - trajectory.value, -config["CLIP_EPS"], config["CLIP_EPS"]
            )
            value_loss = (
                0.5
                * jnp.maximum(
                    (value - target) ** 2, (clipped_value - target) ** 2
                ).mean()
            )
            entropy = (
                -(jax.nn.softmax(logits) * jax.nn.log_softmax(logits)).sum(-1).mean()
            )
            ib = (
                (
                    (free_bit(kl[0]).mean() + free_bit(kl[1]).mean())
                    if rms_scales is None
                    else (
                        normalized_excess(kl[0], rms_scales[0]).mean()
                        + normalized_excess(kl[1], rms_scales[1]).mean()
                    )
                )
                if model.actor.variant in IB_DIVERGENCES
                else jnp.array(0.0)
            )
            total = (
                policy_loss
                + config["VF_COEF"] * value_loss
                - config["ENT_COEF"] * entropy
                + ib_coefficient * ib
            )
            return total, jnp.stack(
                (
                    total,
                    policy_loss,
                    value_loss,
                    entropy,
                    ib,
                    jnp.mean(kl[0]),
                    jnp.mean(kl[1]),
                )
            )

        # Modules contain Python/static fields: partition before entering lax.scan.
        dynamic, static = eqx.partition(network, eqx.is_array)

        def array_epoch(carry, unused):
            dynamic_model, optimizer_state, rng = carry
            model = eqx.combine(dynamic_model, static)
            model, optimizer_state, rng, metrics = epoch_filtered(
                model, optimizer_state, rng
            )
            return (eqx.filter(model, eqx.is_array), optimizer_state, rng), metrics

        def epoch_filtered(model, optimizer_state, rng):
            rng, shuffle_key = jax.random.split(rng)
            permutation = jax.random.permutation(shuffle_key, n).reshape(
                minibatches, n // minibatches
            )

            def array_minibatch(carry, indices):
                dynamic_model, optimizer_state = carry
                model = eqx.combine(dynamic_model, static)
                memory = jax.tree.map(lambda x: x[indices], initial_memory)
                trajectory, advantage, target = jax.tree.map(
                    lambda x: x[:, indices], (batch, advantages, targets)
                )
                (_, metrics), grads = eqx.filter_value_and_grad(loss, has_aux=True)(
                    model, memory, trajectory, advantage, target
                )
                if log_gradients:
                    metrics = jnp.concatenate(
                        (metrics, gradient_norm_metrics(grads, config["MAX_GRAD_NORM"]))
                    )
                changes, optimizer_state = tx.update(
                    grads, optimizer_state, eqx.filter(model, eqx.is_inexact_array)
                )
                model = eqx.apply_updates(model, changes)
                return (eqx.filter(model, eqx.is_array), optimizer_state), metrics

            (dynamic_model, optimizer_state), metrics = jax.lax.scan(
                array_minibatch,
                (eqx.filter(model, eqx.is_array), optimizer_state),
                permutation,
            )
            return (
                eqx.combine(dynamic_model, static),
                optimizer_state,
                rng,
                metrics.mean(0).at[8].set(metrics[:, 8].max())
                if log_gradients
                else metrics.mean(0),
            )

        (dynamic, opt_state, rng), metrics = jax.lax.scan(
            array_epoch,
            (dynamic, opt_state, runner.rng),
            None,
            length=config["UPDATE_EPOCHS"],
        )
        network = eqx.combine(dynamic, static)
        runner = runner._replace(rng=rng)
        completed = batch.next_done.sum()
        return_sum = jnp.where(batch.next_done, batch.episode_return, 0).sum()
        averaged = (
            metrics.mean(0).at[8].set(metrics[:, 8].max())
            if log_gradients
            else metrics.mean(0)
        )
        return network, opt_state, runner, averaged, completed, return_sum

    return update


def episode_horizon(env, params):
    """Respect task-specific limits instead of assuming CartPole's attribute."""
    for name in ("max_steps_in_episode", "max_episode_length"):
        value = getattr(env, name, None)
        if value is not None:
            return int(value)
    value = getattr(params, "max_steps_in_episode", None)
    if value is None:
        raise ValueError("Task does not expose a known evaluation episode bound")
    return int(value)


def evaluate(network, env, params, seed, n, episodes_per_env, max_episode_steps):
    """Equal episode counts per environment; independent held-out reset keys."""
    rng = jax.random.PRNGKey(seed + 1_000_000)
    rng, reset_key = jax.random.split(rng)
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, n), params
    )
    runner = Runner(
        state,
        obs,
        jnp.zeros(n, jnp.int32),
        jnp.zeros(n),
        jnp.ones(n, bool),
        network.initial_state(n),
        rng,
    )

    @eqx.filter_jit
    def run(model, runner):
        def step(carry, _):
            r, counts, scores = carry
            rng, model_key, action_key, env_key = jax.random.split(r.rng, 4)
            memory, logits, _, _ = model.step(
                r.memory,
                r.obs,
                r.previous_action,
                r.previous_reward,
                r.done,
                jax.random.split(model_key, n),
                False,
            )
            action = jax.random.categorical(action_key, logits).astype(jnp.int32)
            obs, state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None)
            )(jax.random.split(env_key, n), r.env_state, action, params)
            take = done & (counts < episodes_per_env)
            scores = scores.at[
                jnp.arange(n), jnp.minimum(counts, episodes_per_env - 1)
            ].add(jnp.where(take, info["returned_episode_returns"], 0))
            counts = counts + take
            r = Runner(state, obs, action, reward, done, memory, rng)
            return r, counts, scores

        def condition(carry):
            t, _, counts, _ = carry
            return (t < episodes_per_env * (max_episode_steps + 1)) & jnp.any(
                counts < episodes_per_env
            )

        def body(carry):
            t, r, counts, scores = carry
            r, counts, scores = step((r, counts, scores), None)
            return t + 1, r, counts, scores

        return jax.lax.while_loop(
            condition,
            body,
            (
                jnp.array(0),
                runner,
                jnp.zeros(n, jnp.int32),
                jnp.zeros((n, episodes_per_env)),
            ),
        )

    _, _, counts, scores = run(network, runner)
    counts, scores = np.asarray(counts), np.asarray(scores)
    if not np.all(counts == episodes_per_env):
        raise RuntimeError(f"Incomplete evaluation: {counts}")
    return scores.ravel()


def write_summary(output):
    results = [
        json.loads(path.read_text()) for path in sorted(output.glob("*/result.json"))
    ]
    rows = []
    for variant in VARIANTS:
        selected = [r for r in results if r["variant"] == variant]
        if not selected:
            continue
        scores = np.array([r["evaluation_mean"] for r in selected])
        biases = []
        for result in selected:
            ref = next(
                (
                    r
                    for r in results
                    if r["variant"] == "mlp" and r["seed"] == result["seed"]
                ),
                None,
            )
            if ref is not None:
                biases.append(ref["evaluation_mean"] - result["evaluation_mean"])
        rows.append(
            {
                "variant": variant,
                "seeds": len(selected),
                "steps_per_seed": selected[0]["actual_steps"],
                "score_mean": float(scores.mean()),
                "score_seed_std": float(scores.std(ddof=1))
                if len(scores) > 1
                else None,
                "memory_bias": float(np.mean(biases)) if biases else None,
            }
        )
    if rows:
        with (output / "summary.csv").open("w") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0])
            writer.writeheader()
            writer.writerows(rows)
        lines = [
            "# Memory-bias pilot results",
            "",
            "Scores are held-out stochastic-policy episode returns. Lower memory bias is better.",
            "",
            "| Variant | Steps/seed | Seeds | Score | MLP − score |",
            "|---|---:|---:|---:|---:|",
        ]
        for row in rows:
            bias = (
                f"{row['memory_bias']:.6f}"
                if row["memory_bias"] is not None
                else "pending"
            )
            lines.append(
                f"| {row['variant']} | {row['steps_per_seed']} | {row['seeds']} | {row['score_mean']:.6f} | {bias} |"
            )
        lines += [
            "",
            "A short pilot with one seed cannot establish a regularization benefit. Full-budget, multiple-seed runs are needed.",
            "",
            "IB heads use current input and posterior mean/log variance, providing a recurrent-state path without sampling.",
        ]
        (output / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return rows


def prepare_ib_calibration(args, variant, seed, config, posterior_input, residual):
    """Resolve the default preset's frozen calibration before allocating training state."""
    if variant not in IB_DIVERGENCES:
        return None, None
    path = getattr(args, "rms_calibration", None)
    normalization = getattr(args, "ib_normalization", None)
    if normalization == "none":
        if path is not None:
            raise ValueError("--ib-normalization none conflicts with --rms-calibration")
        return None, None
    if path is None and normalization is None and variant != DEFAULT_IB_VARIANT:
        return None, None
    if path is None:
        from popgym_arcade.experiments.calibrate_divergences import calibrate

        identity = {
            "env": config["ENV_NAME"],
            "partial": config["PARTIAL"],
            "posterior_input": posterior_input,
            "residual": residual,
            "hidden": args.hidden,
            "layers": args.layers,
            "seed": seed,
            "obs_size": config["OBS_SIZE"],
            "num_envs": config["NUM_ENVS"],
            "num_steps": config["NUM_STEPS"],
        }
        digest = hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:16]
        path = (
            args.output
            / "_calibration"
            / f"{config['ENV_NAME']}_seed{seed}_{digest}.json"
        )
        if not path.exists():
            calibrate(
                config["ENV_NAME"],
                config["PARTIAL"],
                posterior_input,
                seed,
                path,
                hidden=args.hidden,
                layers=args.layers,
                residual=residual,
                config=config,
            )
            jax.clear_caches()
    calibration = json.loads(Path(path).read_text())
    expected = {
        "env": config["ENV_NAME"],
        "partial_obs": config["PARTIAL"],
        "posterior_input": posterior_input,
        "seed": seed,
        "residual": residual,
        "hidden": args.hidden,
        "layers": args.layers,
        "samples": config["NUM_ENVS"] * config["NUM_STEPS"],
    }
    if (
        any(calibration.get(key) != value for key, value in expected.items())
        or calibration.get("obs_size", 128) != config["OBS_SIZE"]
    ):
        raise ValueError(f"RMS calibration does not match run: {path}")
    scales = calibration["scales_bits"][IB_DIVERGENCES[variant]]
    if np.shape(scales) != (2,) or not np.isfinite(scales).all() or min(scales) <= 0:
        raise ValueError("Invalid RMS scales")
    return calibration, scales


def validate_saved_settings(
    metadata, args, variant, seed, config, posterior_input, residual
):
    """Do not reuse a same-named run made with different preset settings."""
    expected = {
        "variant": variant,
        "seed": seed,
        "config": config,
        "hidden": args.hidden,
        "layers": args.layers,
        "requested_steps": args.timesteps,
        "evaluation_episodes": args.eval_episodes,
        "posterior_input": posterior_input,
        "lru_residual": residual,
    }
    schedule_steps = getattr(args, "schedule_timesteps", None)
    if schedule_steps is not None or "learning_rate_schedule_steps" in metadata:
        expected["learning_rate_schedule_steps"] = schedule_steps or args.timesteps
    if variant in IB_DIVERGENCES:
        expected["head_inputs"] = "input_t, posterior_mean, posterior_log_variance"
        normalization = getattr(args, "ib_normalization", None)
        uses_rms = normalization != "none" and (
            getattr(args, "rms_calibration", None) is not None
            or normalization == DEFAULT_IB_NORMALIZATION
            or variant == DEFAULT_IB_VARIANT
        )
        expected["normalization"] = (
            "fixed_calibration_rms_excess_above_one_bit" if uses_rms else None
        )
        expected["ib_coefficient"] = args.ib_coefficient
    mismatched = [key for key, value in expected.items() if metadata.get(key) != value]
    if mismatched:
        raise ValueError(
            f"Saved run differs in {', '.join(mismatched)}; use a fresh output directory"
        )


def run_one(args, variant, seed, config):
    directory = args.output / f"{variant}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    posterior_input, residual = resolve_memory_options(
        variant,
        getattr(args, "posterior_input", None),
        getattr(args, "lru_residual", None),
    )
    if (directory / "result.json").exists():
        validate_saved_settings(
            json.loads((directory / "result.json").read_text()),
            args,
            variant,
            seed,
            config,
            posterior_input,
            residual,
        )
        print(f"Skipping completed {directory}", flush=True)
        return
    started = time.monotonic()
    external_resume = getattr(args, "resume_from", None)
    source_run = None
    source_record = None
    calibration_args = args
    if external_resume is not None:
        external_resume = Path(external_resume)
        candidates = (
            [external_resume / "checkpoint.json"]
            if (external_resume / "checkpoint.json").exists()
            else sorted(
                (external_resume / "checkpoints").glob("step_*/checkpoint.json")
            )
        )
        if not candidates:
            raise ValueError(
                "--resume-from requires a full training checkpoint or run directory"
            )
        source_record = json.loads(candidates[-1].read_text())
        source_run = candidates[-1].parent.parent.parent
        if source_run.resolve() == directory.resolve():
            raise ValueError("--resume-from must use a fresh output directory")
        previous_calibration = source_record["run_metadata"].get("rms_calibration")
        if (
            previous_calibration is not None
            and getattr(args, "rms_calibration", None) is None
        ):
            frozen = args.output / "_calibration" / f"resumed_seed{seed}.json"
            frozen.parent.mkdir(parents=True, exist_ok=True)
            frozen.write_text(json.dumps(previous_calibration, indent=2) + "\n")
            calibration_args = argparse.Namespace(**vars(args))
            calibration_args.rms_calibration = frozen
    calibration, rms_scales = prepare_ib_calibration(
        calibration_args, variant, seed, config, posterior_input, residual
    )
    network = MemoryActorCritic(
        variant,
        jax.random.PRNGKey(seed),
        hidden=args.hidden,
        layers=args.layers,
        dropout=args.dropout,
        obs_size=config["OBS_SIZE"],
        posterior_input=posterior_input,
        residual=residual,
    )
    env, params = popgym_arcade.make(
        config["ENV_NAME"], partial_obs=config["PARTIAL"], obs_size=config["OBS_SIZE"]
    )
    env = LogWrapper(env)
    n, steps = config["NUM_ENVS"], config["NUM_STEPS"]
    checkpoint = directory / "model.eqx"
    if checkpoint.exists() and (directory / "config.json").exists():
        # Training finished before checkpoint creation; recover only evaluation.
        metadata = json.loads((directory / "config.json").read_text())
        validate_saved_settings(
            metadata, args, variant, seed, config, posterior_input, residual
        )
        records = [
            json.loads(line)
            for line in (directory / "training.jsonl").read_text().splitlines()
        ]
        expected_steps = int(args.timesteps // (n * steps)) * n * steps
        if (
            not records
            or records[-1]["steps"] != expected_steps
            or metadata["actual_steps"] != expected_steps
            or metadata["config"]["ENV_NAME"] != config["ENV_NAME"]
            or metadata["seed"] != seed
        ):
            raise ValueError(f"Checkpoint does not match requested run: {directory}")
        network = eqx.tree_deserialise_leaves(checkpoint, network)
        scores = evaluate(
            network,
            env,
            params,
            seed,
            n,
            args.eval_episodes // n,
            episode_horizon(env, params),
        )
        np.save(directory / "evaluation_returns.npy", scores)
        result = {
            **metadata,
            "evaluation_mean": float(scores.mean()),
            "evaluation_std": float(scores.std(ddof=1)),
            "evaluation_standard_error": float(
                scores.std(ddof=1) / np.sqrt(len(scores))
            ),
            "seconds": records[-1]["seconds"] + time.monotonic() - started,
            "evaluation_recovered_from_checkpoint": True,
            "evaluation_code_sha256": hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2))
        print("RECOVERED EVALUATION " + json.dumps(result), flush=True)
        write_summary(args.output)
        jax.clear_caches()
        return
    num_updates = int(args.timesteps // (n * steps))
    if num_updates < 1:
        raise ValueError("Need at least one rollout of training")
    schedule_steps = getattr(args, "schedule_timesteps", None) or args.timesteps
    schedule_updates = max(1, int(schedule_steps // (n * steps)))
    schedule = optax.linear_schedule(
        config["LR"],
        1e-10,
        schedule_updates * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"],
    )
    optimizer = (
        optax.adamw(schedule, eps=1e-5, weight_decay=args.weight_decay)
        if variant == "weight_decay"
        else optax.adam(schedule, eps=1e-5)
    )
    tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optimizer)
    opt_state = tx.init(eqx.filter(network, eqx.is_inexact_array))
    rng, reset_key = jax.random.split(jax.random.PRNGKey(seed + 10_000))
    obs, state = jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(reset_key, n), params
    )
    runner = Runner(
        state,
        obs,
        jnp.zeros(n, jnp.int32),
        jnp.zeros(n),
        jnp.ones(n, bool),
        network.initial_state(n),
        rng,
    )
    metadata = {
        "head_inputs": (
            "input_t, posterior_mean, posterior_log_variance"
            if variant in IB_DIVERGENCES
            else "input_t"
            if variant == "mlp"
            else "input_t, projected_lru_output"
        ),
        "normalization": "fixed_calibration_rms_excess_above_one_bit"
        if calibration
        else None,
        "rms_calibration": calibration,
        "rms_scales_bits": rms_scales,
        "variant": variant,
        "seed": seed,
        "upstream_commit": "d061b611718ae55d095791b4ea7046b5266cafd4",
        "config": config,
        "requested_steps": args.timesteps,
        "actual_steps": num_updates * n * steps,
        "weight_decay": args.weight_decay if variant == "weight_decay" else 0,
        "state_dropout": args.dropout if variant == "state_dropout" else 0,
        "ib_coefficient": args.ib_coefficient if variant in IB_DIVERGENCES else 0,
        "ib_kl_units": "bits" if variant != "ib_renyi_half" else None,
        "ib_divergence_units": "bits",
        "divergence": IB_DIVERGENCES.get(variant),
        "kl_prior_gradient_weight": 2 * KL_BALANCES.get(variant, 0.5),
        "kl_posterior_gradient_weight": 2 * (1 - KL_BALANCES.get(variant, 0.5)),
        "posterior_input": posterior_input,
        "lru_residual": residual,
        "ib_free_bits": 1,
        "ib_distribution": "diagonal multivariate normal",
        "hidden": args.hidden,
        "layers": args.layers,
        "flattened_state_real_dimension_per_branch": 2 * args.hidden * args.layers,
        "gaussian_dimension_per_branch": network.actor.latent_dim,
        "device": str(jax.devices()[0]),
        "device_kind": jax.devices()[0].device_kind,
        "parameter_count": sum(
            x.size for x in jax.tree.leaves(eqx.filter(network, eqx.is_inexact_array))
        ),
        "source_sha256": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__),
                Path(__file__).with_name("memory_models.py"),
                Path(__file__).with_name("calibrate_divergences.py"),
                Path(__file__).with_name("checkpoints.py"),
                Path("uv.lock"),
            )
        },
        "jax_version": jax.__version__,
        "evaluation_episodes": args.eval_episodes,
        "learning_rate_schedule_steps": schedule_steps,
        "gradient_log_interval": getattr(args, "gradient_log_interval", 0),
        "resume_from": str(external_resume) if external_resume is not None else None,
        "budget_extension": bool(getattr(args, "extend_training", False)),
        "extension_schedule_policy": (
            "preserved horizon and optimizer counter"
            if schedule_steps
            == source_record["run_metadata"].get("learning_rate_schedule_steps")
            else "new horizon with preserved optimizer counter"
        )
        if source_record is not None
        else None,
        "checkpoint_interval": getattr(args, "checkpoint_interval", 0),
        "checkpoint_state": "full_training_state"
        if getattr(args, "checkpoint_interval", 0)
        else None,
    }
    start_update = 0
    previous_elapsed = 0.0
    local_checkpoint = bool(
        list((directory / "checkpoints").glob("step_*/checkpoint.json"))
    )
    resume_location = (
        directory if local_checkpoint or external_resume is None else external_resume
    )
    restored = restore_checkpoint(
        resume_location,
        (network, opt_state, runner),
        metadata,
        n * steps,
        allow_budget_extension=bool(getattr(args, "extend_training", False)),
    )
    if restored is not None:
        (network, opt_state, runner), saved = restored
        start_update = saved["completed_updates"]
        previous_elapsed = saved["elapsed_seconds"]
        if resume_location != directory and source_run is not None:
            shutil.copy2(source_run / "training.jsonl", directory / "training.jsonl")
            if (source_run / "gradient_norms.jsonl").exists():
                shutil.copy2(
                    source_run / "gradient_norms.jsonl",
                    directory / "gradient_norms.jsonl",
                )
        trim_training_log(directory / "training.jsonl", start_update)
        gradient_log = directory / "gradient_norms.jsonl"
        if gradient_log.exists():
            kept = []
            for line in gradient_log.read_text().splitlines():
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    break
                if row["steps"] <= saved["actual_steps"]:
                    kept.append(row)
            gradient_log.write_text("".join(json.dumps(row) + "\n" for row in kept))
        print(
            f"RESUME {variant} seed={seed} update={start_update} steps={saved['actual_steps']}",
            flush=True,
        )
    (directory / "config.json").write_text(json.dumps(metadata, indent=2))
    gradient_interval = getattr(args, "gradient_log_interval", 0)
    update = build_update(
        env,
        params,
        config,
        tx,
        args.ib_coefficient,
        rms_scales,
        log_gradients=bool(gradient_interval),
    )
    print(
        f"START {variant} seed={seed} updates={num_updates} device={jax.devices()[0]}",
        flush=True,
    )
    with (directory / "training.jsonl").open("a" if start_update else "w") as log:
        for index in range(start_update, num_updates):
            network, opt_state, runner, metrics, completed, return_sum = update(
                network, opt_state, runner
            )
            metrics = np.asarray(metrics)
            if not np.isfinite(metrics).all():
                raise FloatingPointError(
                    f"Nonfinite training metrics: {variant} update={index} {metrics}"
                )
            completed = int(completed)
            record = {
                "update": index + 1,
                "steps": (index + 1) * n * steps,
                "seconds": previous_elapsed + time.monotonic() - started,
                "loss": float(metrics[0]),
                "policy_loss": float(metrics[1]),
                "value_loss": float(metrics[2]),
                "entropy": float(metrics[3]),
                "ib_loss_bits": float(metrics[4]) if rms_scales is None else None,
                "ib_loss_normalized": float(metrics[4])
                if rms_scales is not None
                else None,
                "actor_divergence_nats": float(metrics[5]),
                "critic_divergence_nats": float(metrics[6]),
                "episodes": completed,
                "training_return": float(return_sum) / completed if completed else None,
            }
            if gradient_interval:
                record.update(
                    {
                        "gradient_norm_pre_clip_mean": float(metrics[7]),
                        "gradient_norm_pre_clip_max": float(metrics[8]),
                        "gradient_norm_post_clip_mean": float(metrics[9]),
                        "actor_gradient_norm_mean": float(metrics[10]),
                        "critic_gradient_norm_mean": float(metrics[11]),
                    }
                )
                if (
                    checkpoint_due(
                        index * n * steps, (index + 1) * n * steps, gradient_interval
                    )
                    or index + 1 == num_updates
                ):
                    with (directory / "gradient_norms.jsonl").open("a") as gradient_log:
                        gradient_log.write(
                            json.dumps(
                                {
                                    **record,
                                    "sampled_rollout_steps": n * steps,
                                    "ppo_minibatches_in_sample": config["UPDATE_EPOCHS"]
                                    * config["NUM_MINIBATCHES"],
                                    "clip_threshold": config["MAX_GRAD_NORM"],
                                }
                            )
                            + "\n"
                        )
            log.write(json.dumps(record) + "\n")
            log.flush()
            interval = getattr(args, "checkpoint_interval", 0)
            if checkpoint_due(index * n * steps, (index + 1) * n * steps, interval):
                saved_path = save_checkpoint(
                    directory,
                    (network, opt_state, runner),
                    metadata,
                    index + 1,
                    (index + 1) * n * steps,
                    record["seconds"],
                    interval,
                    final=index + 1 == num_updates,
                )
                print(f"CHECKPOINT {saved_path}", flush=True)
            if index == 0 or (index + 1) % 10 == 0 or index + 1 == num_updates:
                print(json.dumps({"variant": variant, **record}), flush=True)
    if getattr(args, "checkpoint_interval", 0):
        saved_path = save_checkpoint(
            directory,
            (network, opt_state, runner),
            metadata,
            num_updates,
            num_updates * n * steps,
            previous_elapsed + time.monotonic() - started,
            args.checkpoint_interval,
            final=True,
        )
        print(f"FINAL CHECKPOINT {saved_path}", flush=True)
    temporary_model = directory / ".model.eqx.tmp"
    with temporary_model.open("wb") as stream:
        eqx.tree_serialise_leaves(stream, network)
    temporary_model.replace(directory / "model.eqx")
    scores = evaluate(
        network,
        env,
        params,
        seed,
        n,
        args.eval_episodes // n,
        episode_horizon(env, params),
    )
    np.save(directory / "evaluation_returns.npy", scores)
    result = {
        **metadata,
        "evaluation_mean": float(scores.mean()),
        "evaluation_std": float(scores.std(ddof=1)),
        "evaluation_standard_error": float(scores.std(ddof=1) / np.sqrt(len(scores))),
        "seconds": previous_elapsed + time.monotonic() - started,
    }
    (directory / "result.json").write_text(json.dumps(result, indent=2))
    print("RESULT " + json.dumps(result), flush=True)
    print(json.dumps(write_summary(args.output)), flush=True)
    jax.clear_caches()


def build_parser(defaults=None):
    defaults = upstream_config() if defaults is None else defaults
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timesteps", type=int, default=int(defaults["TOTAL_TIMESTEPS"])
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument(
        "--variants", choices=VARIANTS, nargs="+", default=list(DEFAULT_VARIANTS)
    )
    parser.add_argument(
        "--posterior-input",
        choices=["state", "output"],
        default=None,
        help="IB default: output; explicit state selects the full recurrent state",
    )
    parser.add_argument(
        "--lru-residual",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="IB default: enabled; vanilla controls default: disabled",
    )
    parser.add_argument(
        "--ib-normalization",
        choices=[DEFAULT_IB_NORMALIZATION, "none"],
        default=None,
        help="Default IB preset: fixed_rms with automatic calibration; other variants: none",
    )
    parser.add_argument("--rms-calibration", type=Path)
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=0,
        help="Save resumable training state at rollout boundaries crossing this many steps; 0 disables periodic saves",
    )
    parser.add_argument(
        "--gradient-log-interval",
        type=int,
        default=0,
        help="Sample pre/post clipping and actor/critic gradient norms at this many environment steps",
    )
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Resume full state from a run or checkpoint directory into a fresh output directory",
    )
    parser.add_argument(
        "--extend-training",
        action="store_true",
        help="Explicitly permit a larger total budget when using --resume-from",
    )
    parser.add_argument(
        "--schedule-timesteps",
        type=int,
        default=None,
        help="LR schedule horizon; defaults to the requested total budget, with optimizer counters preserved on resume",
    )
    parser.add_argument("--partial-obs", action="store_true")
    parser.add_argument("--env", default=defaults["ENV_NAME"])
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--ib-coefficient", type=float, default=1.0)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--output", type=Path, default=Path("results/memory_bias"))
    return parser


def main():
    defaults = upstream_config()
    parser = build_parser(defaults)
    args = parser.parse_args()
    if args.ib_normalization == "none" and args.rms_calibration is not None:
        parser.error("--ib-normalization none conflicts with --rms-calibration")
    if args.resume_from is not None and (
        len(args.variants) != 1 or len(args.seeds) != 1
    ):
        parser.error("--resume-from requires exactly one variant and seed")
    if args.extend_training and args.resume_from is None:
        parser.error(
            "--extend-training requires --resume-from and a fresh output directory"
        )
    if args.gradient_log_interval < 0 or (
        args.schedule_timesteps is not None and args.schedule_timesteps < 1
    ):
        parser.error(
            "Gradient interval must be nonnegative and schedule horizon positive"
        )
    if args.checkpoint_interval < 0:
        parser.error("--checkpoint-interval must be nonnegative")
    if not 0 <= args.dropout < 1:
        parser.error("--dropout must be in [0, 1)")
    if (
        args.hidden < 1
        or args.layers < 1
        or args.eval_episodes < defaults["NUM_ENVS"]
        or args.eval_episodes % defaults["NUM_ENVS"]
    ):
        parser.error(
            "Positive model dimensions and an evaluation episode count divisible by NUM_ENVS are required"
        )
    if jax.default_backend() != "gpu":
        raise RuntimeError("This experiment requires a CUDA GPU")
    config = defaults | {
        "ENV_NAME": args.env,
        "PARTIAL": args.partial_obs,
        "TOTAL_TIMESTEPS": args.timesteps,
        "WANDB_MODE": "disabled",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for variant in args.variants:
            run_one(args, variant, seed, config)


if __name__ == "__main__":
    main()
