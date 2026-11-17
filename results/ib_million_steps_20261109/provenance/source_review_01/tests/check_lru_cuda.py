"""Manual full-size CUDA equivalence check; not part of CPU pytest collection."""

import json
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from popgym_arcade.experiments.memory_models import Branch


def main():
    if jax.default_backend() != "gpu":
        raise RuntimeError("Run this check with JAX_PLATFORMS=cuda")
    branch = Branch(
        eqx.nn.Sequential([]), "state_dropout", 5, 512, 2, 0.1, jax.random.PRNGKey(21)
    )
    inputs = jax.random.normal(jax.random.PRNGKey(22), (128, 518)) * 0.1
    state = jax.random.normal(jax.random.PRNGKey(23), (2, 512)).astype(
        jnp.complex64
    ) * (0.1 + 0.2j)
    done = jnp.arange(128) == 63
    keys = jax.random.split(jax.random.PRNGKey(24), 128)

    @eqx.filter_jit
    def check(branch):
        def step(state, xs):
            x, d, k = xs
            state, logits, _kl = branch.step(state, x, d, k, True)
            return state, logits

        serial_state, serial_logits = jax.lax.scan(step, state, (inputs, done, keys))
        parallel_state, parallel_logits, _ = branch.parallel_sequence(
            state, inputs, done, keys, True
        )
        return serial_state, serial_logits, parallel_state, parallel_logits

    ss, sl, ps, pl = map(np.asarray, check(branch))
    np.testing.assert_allclose(ss, ps, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(sl, pl, rtol=1e-4, atol=1e-4)
    result = {
        "device": jax.devices()[0].device_kind,
        "sequence_length": 128,
        "hidden": 512,
        "layers": 2,
        "max_absolute_state_difference": float(np.max(np.abs(ss - ps))),
        "max_absolute_logit_difference": float(np.max(np.abs(sl - pl))),
        "passed": True,
    }
    output = Path("results/development_checks/parallel_cuda.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    print(result)


if __name__ == "__main__":
    main()
