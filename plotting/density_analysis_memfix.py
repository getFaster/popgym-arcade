"""Upstream PPO recall definition applied to MEMFIX .eqx checkpoints.

Greedy rollout, gradient of summed Categorical.logits, absolute pixel L1,
per-trajectory normalization (including upstream's zero convention).
"""

import argparse
import csv
import hashlib
import json
import os
import resource
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from distreqx import distributions

import popgym_arcade
from plotting.density_analysis_ppo import normalize_recall_gradients
from popgym_arcade.experiments.memory_models import MemoryActorCritic


def bin_distribution(distribution, bins=16):
    """Sum mass in equal normalized-time intervals; include tau=1 in last."""
    positions = np.linspace(0, 1, len(distribution))
    indices = np.minimum((positions * bins).astype(int), bins - 1)
    return np.bincount(indices, weights=distribution, minlength=bins)


@eqx.filter_jit
def actor_step(actor, memory, obs, action, reward, done):
    x = actor.encode(obs[None], action[None], reward[None], done[None])[0]
    memory, logits, _ = actor.step(memory, x, done, jax.random.key(0), False)
    return memory, logits


@eqx.filter_jit
def recall(
    actor,
    obs,
    action,
    reward,
    done,
    definition="upstream",
    endpoint=None,
    initial_state=None,
):
    def target(images):
        def body(memory, inputs):
            memory, logits = actor_step(actor, memory, *inputs)
            return memory, logits

        initial = (
            jnp.zeros((actor.layers, actor.hidden), jnp.complex64)
            if initial_state is None
            else initial_state
        )
        _, logits = jax.lax.scan(body, initial, (images, action, reward, done))
        # Match policy.logits in upstream get_gradient_ppo (normalized logits).
        index = len(obs) - 1 if endpoint is None else endpoint
        if definition == "paper":
            return jax.nn.softmax(logits[index])
        return distributions.Categorical(logits=logits[index]).logits.sum()

    if definition == "paper":
        # Action-wise absolute gradients must precede summation across actions.
        probabilities, pullback = jax.vjp(target, obs.astype(jnp.float32))

        # Sequential action VJPs avoid materializing the action-by-pixel Jacobian.
        def accumulate(mass, basis):
            gradient = pullback(basis)[0]
            return mass + jnp.abs(gradient).sum(axis=(1, 2, 3)), None

        mass, _ = jax.lax.scan(
            accumulate, jnp.zeros(len(obs)), jnp.eye(len(probabilities))
        )
        total = mass.sum()
        return jnp.where(total > 0, mass / total, jnp.zeros_like(mass))
    else:
        grads = jax.grad(target)(obs.astype(jnp.float32))
    return normalize_recall_gradients(grads)


def load(path, metadata):
    m = metadata
    model = MemoryActorCritic(
        m["variant"],
        jax.random.PRNGKey(m["seed"]),
        hidden=m["hidden"],
        layers=m["layers"],
        dropout=m.get("state_dropout", 0.1),
        obs_size=m["config"]["OBS_SIZE"],
        posterior_input=m.get("posterior_input"),
        residual=m.get("lru_residual"),
        recurrent_state_quantization=m.get("recurrent_state_quantization", "none"),
        fsq_levels=m.get("fsq_levels", 5),
        fsq_scale=m.get("fsq_scale", 1.0),
    )
    return eqx.tree_deserialise_leaves(path, model)


def collect(model, env, params, seed, steps, window=10):
    """Roll forward without gradients; retain only the last window and its carry."""
    from collections import deque

    rng, key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = env.reset(key, params)
    memory = model.initial_state(1)[0][0]
    action, reward, done = jnp.array(0), jnp.array(0.0), jnp.array(True)
    history, carries = deque(maxlen=window), deque(maxlen=window)
    for timestep in range(steps):
        history.append((obs, action, reward, done))
        carries.append(memory)
        memory, logits = actor_step(model.actor, memory, obs, action, reward, done)
        action = logits.argmax().astype(jnp.int32)
        rng, key = jax.random.split(rng)
        obs, state, reward, terminal, _ = step_env(env, key, state, action, params)
        done = jnp.array(False)
        if bool(terminal):
            break
    return (tuple(jnp.stack(x) for x in zip(*history)), carries[0], timestep + 1)


def collect_episode(model, env, params, seed, safety_limit):
    """Collect every policy input up to termination, never an autoreset input.

    A safety limit is a failure boundary, not an artificial episode endpoint.
    """
    rng, key = jax.random.split(jax.random.PRNGKey(seed))
    obs, state = env.reset(key, params)
    memory = model.initial_state(1)[0][0]
    action, reward, done = jnp.array(0), jnp.array(0.0), jnp.array(True)
    history = []
    for _ in range(safety_limit):
        history.append((obs, action, reward, done))
        memory, logits = actor_step(model.actor, memory, obs, action, reward, done)
        action = logits.argmax().astype(jnp.int32)
        rng, key = jax.random.split(rng)
        obs, state, reward, terminal, _ = step_env(env, key, state, action, params)
        done = jnp.array(False)
        if bool(terminal):
            return tuple(jnp.stack(x) for x in zip(*history))
    raise RuntimeError(f"Episode did not terminate within {safety_limit} steps")


@eqx.filter_jit
def step_env(env, key, state, action, params):
    return env.step(key, state, action, params)


def family(m):
    if (
        m["variant"] == "lru"
        and m.get("recurrent_state_quantization", "none") == "none"
    ):
        return "Plain LRU"
    if m.get("recurrent_state_quantization", "none") != "none":
        return "FSQ"
    if "prediction" in m["variant"] and not m["variant"].startswith("mlp"):
        return "Prediction"
    if m["variant"].startswith("ib_") or m["variant"] == "information_bottleneck":
        return (
            "KL" if m.get("divergence") not in ("renyi_half",) else "Other divergence"
        )
    return None


def choose_endpoint(requested, collected, window=10, fallback=False):
    """Never use an autoreset observation or a window shorter than requested."""
    if collected < window:
        return None, "insufficient_history"
    if collected >= requested + 1:
        return requested, "defined"
    if fallback:
        return collected - 1, "defined_terminal_fallback"
    return None, "terminated_before_endpoint"


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=Path("results"))
    parser.add_argument(
        "--output", type=Path, default=Path("results/recall_density_16")
    )
    parser.add_argument(
        "--definition", choices=["upstream", "paper"], default="upstream"
    )
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--match", default="")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--endpoint-plan", type=Path)
    parser.add_argument("--terminal-fallback", action="store_true")
    parser.add_argument("--reuse-manifest", type=Path)
    args = parser.parse_args()
    if args.samples < 1 or args.max_steps < 1:
        parser.error("samples and max-steps must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = []
    plan = json.loads(args.endpoint_plan.read_text()) if args.endpoint_plan else None
    if plan and (
        len(plan["endpoints"]) != args.samples or plan["window"] != args.max_steps
    ):
        parser.error("endpoint plan does not match samples/window")
    reuse = (
        {r["checkpoint"]: r for r in json.loads(args.reuse_manifest.read_text())}
        if args.reuse_manifest
        else {}
    )
    for path in sorted(args.root.rglob("model.eqx")):
        if "provenance" in path.parts or args.match not in str(path):
            continue
        if args.checkpoint and path.resolve() != args.checkpoint.resolve():
            continue
        config_path = path.with_name("config.json")
        if not config_path.exists():
            continue
        m = json.loads(config_path.read_text())
        intervention = family(m)
        if intervention is None:
            continue
        row = {
            "terminal_fallback": args.terminal_fallback,
            "endpoint_plan": plan,
            "definition": args.definition,
            "checkpoint": str(path),
            "task": m["config"]["ENV_NAME"],
            "mode": "POMDP" if m["config"]["PARTIAL"] else "MDP",
            "intervention": intervention,
            "variant": m["variant"],
            "training_seed": m["seed"],
            "actual_steps": m["actual_steps"],
            "quantization": m.get("recurrent_state_quantization", "none"),
            "levels": m.get("fsq_levels"),
            "posterior_input": m.get("posterior_input"),
            "residual": m.get("lru_residual"),
            "samples": args.samples,
            "max_steps": args.max_steps,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "head_inputs": m.get("head_inputs"),
            "divergence": m.get("divergence"),
            "source_sha256": m.get("source_sha256"),
            "device": str(jax.devices()[0]),
            "resource_limits": {
                "cpu_affinity": sorted(os.sched_getaffinity(0)),
                "address_space_bytes": resource.getrlimit(resource.RLIMIT_AS)[0],
            },
        }
        token = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[
            :20
        ]
        result_path = args.output / f"{token}.json"
        if result_path.exists():
            stored = json.loads(result_path.read_text())
            if stored["status"] in ("complete", "undefined"):
                manifest.append(stored)
                continue
        print(f"Analyzing {path}", flush=True)
        try:
            model = load(path, m)
            env, params = popgym_arcade.make(
                m["config"]["ENV_NAME"],
                partial_obs=m["config"]["PARTIAL"],
                obs_size=m["config"]["OBS_SIZE"],
            )
            densities, lengths, distributions = [], [], []
            sample_status, endpoints = [], []
            requested_endpoints = []
            cached = reuse.get(str(path))
            if cached and not (
                cached.get("sha256") == row["sha256"]
                and cached.get("endpoint_plan") == plan
                and cached.get("definition") == args.definition
            ):
                cached = None
            for sample in range(args.samples):
                endpoint = plan["endpoints"][sample] if plan else args.max_steps - 1
                seed = plan["trajectory_seeds"][sample] if plan else 1_000_000 + sample
                requested_endpoints.append(endpoint)
                cached_dist = cached["distributions"][sample] if cached else None
                collected = cached["lengths"][sample] if cached else None
                if cached_dist is not None:
                    dist = np.asarray(cached_dist)
                    actual_endpoint = cached["endpoints"][sample]
                    status = "defined_reused"
                elif collected is not None and (
                    collected < args.max_steps or not args.terminal_fallback
                ):
                    endpoints.append(None)
                    lengths.append(collected)
                    distributions.append(None)
                    sample_status.append(
                        "insufficient_history"
                        if collected < args.max_steps
                        else "terminated_before_endpoint"
                    )
                    continue
                else:
                    history, carry, collected = collect(
                        model, env, params, seed, endpoint + 1, window=args.max_steps
                    )
                    actual_endpoint, status = choose_endpoint(
                        endpoint, collected, args.max_steps, args.terminal_fallback
                    )
                    if actual_endpoint is None:
                        endpoints.append(None)
                        lengths.append(collected)
                        distributions.append(None)
                        sample_status.append(status)
                        continue
                    dist = np.asarray(
                        recall(
                            model.actor,
                            *history,
                            definition=args.definition,
                            initial_state=carry,
                        )
                    )
                endpoints.append(actual_endpoint)
                lengths.append(collected)
                distributions.append(dist.tolist())
                sample_status.append(
                    status if dist.sum() > 0 else "zero_gradient_undefined"
                )
                if dist.sum() > 0:
                    densities.append(bin_distribution(dist))
            values = np.asarray(densities).reshape(-1, 16)
            row.update(
                status="complete" if len(values) else "undefined",
                endpoints=endpoints,
                requested_endpoints=requested_endpoints,
                window_starts=[
                    t - args.max_steps + 1 if t is not None else None for t in endpoints
                ],
                attribution_scope="window-conditional: pre-window carry held fixed; normalization over 10 observations",
                sample_status=sample_status,
                valid_samples=len(values),
                lengths=lengths,
                distributions=distributions,
                bin_mean=values.mean(0).tolist() if len(values) else [],
                bin_se=(values.std(0, ddof=1) / np.sqrt(len(values))).tolist()
                if len(values) > 1
                else None,
                zero_samples=sample_status.count("zero_gradient_undefined"),
            )
        except Exception as exc:  # noqa: BLE001 -- retain per-checkpoint failures in audit
            row.update(status="error", error=str(exc))
            print(f"ERROR {exc}", flush=True)
        row["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        result_path.write_text(json.dumps(row, indent=2) + "\n")
        manifest.append(row)
        write_report(args.output, manifest)
    write_report(args.output, manifest)


def write_report(output, rows):
    (output / "manifest.json").write_text(json.dumps(rows, indent=2) + "\n")
    fields = [
        "checkpoint",
        "task",
        "mode",
        "intervention",
        "variant",
        "training_seed",
        "actual_steps",
        "status",
    ] + [f"bin_{i + 1:02d}" for i in range(16)]
    with (output / "recall_16.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            entry = {k: row.get(k) for k in fields[:8]}
            entry.update(
                {f"bin_{i + 1:02d}": v for i, v in enumerate(row.get("bin_mean", []))}
            )
            writer.writerow(entry)
    complete = sum(r["status"] == "complete" for r in rows)
    paper = bool(rows and rows[0].get("definition") == "paper")
    (output / "REPORT.md").write_text(
        "# MEMFIX recall density: 16 bins\n\n"
        f"{complete}/{len(rows)} checkpoints completed. See [all checkpoint bins](recall_16.csv) "
        "and [sample distributions, standard errors, metadata and errors](manifest.json).\n\n"
        "Uses plotting/density_analysis_ppo.py: greedy actions; gradient of the sum of "
        "normalized Categorical logits, then absolute pixel L1 and per-trajectory normalization. "
        "Zero gradients retain upstream zero distributions. This is not the sum of separate "
        "action-probability gradient norms. Previous action/reward are fixed during replay. "
        "Episodes start with zero action/reward and reset memory. No post-terminal observation is replayed. "
        "Replay uses sequential recurrence with no detached memory. FSQ uses its existing straight-through gradient.\n\n"
        "Endpoints and window boundaries are recorded for every sample. "
        "These are short-prefix diagnostics, not full-episode recall. With at most 10 observations, "
        "some of 16 bins are structurally empty. Bins are equal intervals of t/(length-1), "
        "with the right endpoint included in bin 16. Values are probability masses, not density heights. "
        "SEs describe reset/transition samples conditional on each checkpoint; training seeds remain separate. "
        "No training or original artifacts are modified. Configurations with missing checkpoints cannot be evaluated.\n"
    )

    if paper:
        report = output / "REPORT.md"
        text = report.read_text()
        start = text.index("Uses plotting/")
        end = text.index("Previous action/reward")
        text = (
            text[:start]
            + "Paper policy-probability definition: differentiate each action probability separately; take absolute pixel gradients, sum over pixels and actions, then normalize across time. Zero totals are retained as zero rows and counted explicitly. "
            + text[end:]
        )
        report.write_text(text)
    if rows and rows[0].get("endpoint_plan"):
        report = output / "REPORT.md"
        text = report.read_text()
        start = text.index("Endpoints and window boundaries")
        end = text.index("SEs describe", start)
        text = (
            text[:start]
            + (
                "Ten-observation sliding windows end at the shared sampled endpoints recorded in the manifest. "
                "The actual pre-window recurrent carry is held fixed: recall is conditional on that carry and "
                "normalized within the window, not across the entire episode. Positions are (j-(t-9))/9; "
                "16 equal-width bins include 1 in the last bin. Zero-gradient samples are undefined and excluded. "
                "Episodes ending before the assigned endpoint are unavailable, with no resampling. "
            )
            + text[end:]
        )
        report.write_text(text)


if __name__ == "__main__":
    main()
