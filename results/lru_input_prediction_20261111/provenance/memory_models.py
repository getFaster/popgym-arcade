"""LRU controls for the fully observed memory-bias experiment."""

import equinox as eqx
import jax
import jax.numpy as jnp
from equinox import nn

from popgym_arcade.baselines.model.builder import ActorCritic

DEFAULT_IB_VARIANT = "ib_reverse_prior20"
DEFAULT_IB_POSTERIOR_INPUT = "output"
DEFAULT_IB_RESIDUAL = True
DEFAULT_IB_NORMALIZATION = "fixed_rms"

ORIGINAL_VARIANTS = (
    "mlp",
    "lru",
    "weight_decay",
    "state_dropout",
    "information_bottleneck",
)
DEFAULT_VARIANTS = tuple(
    DEFAULT_IB_VARIANT if variant == "information_bottleneck" else variant
    for variant in ORIGINAL_VARIANTS
)


def resolve_memory_options(variant, posterior_input=None, residual=None):
    """IB uses residual output by default; vanilla controls retain their defaults."""
    is_ib = variant in IB_DIVERGENCES
    if posterior_input is None:
        posterior_input = DEFAULT_IB_POSTERIOR_INPUT if is_ib else "state"
    if residual is None:
        residual = DEFAULT_IB_RESIDUAL if is_ib else False
    if posterior_input not in ("state", "output"):
        raise ValueError("posterior_input must be state or output")
    return posterior_input, residual


IB_DIVERGENCES = {
    "information_bottleneck": "forward_kl",
    "ib_reverse_kl": "reverse_kl",
    "ib_symmetric_kl": "symmetric_kl",
    "ib_renyi_half": "renyi_half",
}
# Prior/posterior gradient weights sum to two, preserving the original
# (1, 1) gradient at alpha=0.5 rather than halving the auxiliary loss.
KL_BALANCES = {
    f"ib_{direction}_prior{int(alpha * 100)}": alpha
    for direction in ("forward", "reverse", "symmetric", "renyi_half")
    for alpha in (0.2, 0.8)
}
IB_DIVERGENCES.update(
    {
        name: ("renyi_half" if "renyi_half" in name else name.split("_")[1] + "_kl")
        for name in KL_BALANCES
    }
)
IB_DIVERGENCES["ib_reverse_prior20_prediction"] = "reverse_kl"
KL_BALANCES["ib_reverse_prior20_prediction"] = 0.2

VARIANTS = (
    ORIGINAL_VARIANTS
    + ("lru_prediction",)
    + tuple(k for k in IB_DIVERGENCES if k not in ORIGINAL_VARIANTS)
)


def diagonal_kl(prior_mean, prior_logvar, posterior_mean, posterior_logvar):
    """KL(prior || posterior), summed over the multivariate event (nats)."""
    return 0.5 * jnp.sum(
        posterior_logvar
        - prior_logvar
        + jnp.exp(prior_logvar - posterior_logvar)
        + jnp.square(prior_mean - posterior_mean) * jnp.exp(-posterior_logvar)
        - 1,
        axis=-1,
    )


def gaussian_divergence(pm, pl, qm, ql, kind):
    """Exact full-event divergence for diagonal Gaussians, in natural log units.

    Rényi-1/2 = -2 log integral(sqrt(p q)) = twice Bhattacharyya distance.
    All choices are unbounded and can use the existing one-bit free region.
    """
    if kind == "forward_kl":
        return diagonal_kl(pm, pl, qm, ql)
    if kind == "reverse_kl":
        return diagonal_kl(qm, ql, pm, pl)
    if kind == "symmetric_kl":
        return 0.5 * (diagonal_kl(pm, pl, qm, ql) + diagonal_kl(qm, ql, pm, pl))
    if kind == "renyi_half":
        log_average_variance = jnp.logaddexp(pl, ql) - jnp.log(2.0)
        mean_term = 0.25 * jnp.square(pm - qm) * jnp.exp(-log_average_variance)
        variance_term = log_average_variance - 0.5 * (pl + ql)
        return jnp.maximum(0.0, jnp.sum(mean_term + variance_term, axis=-1))
    raise ValueError(f"Unknown Gaussian divergence: {kind}")


def balanced_divergence(pm, pl, qm, ql, kind, alpha=0.5):
    """Original divergence value; prior/posterior gradients weighted 2a/2(1-a)."""
    if not 0 <= alpha <= 1:
        raise ValueError("Prior balance must be in [0, 1]")
    if alpha == 0.5:
        return gaussian_divergence(pm, pl, qm, ql, kind)
    sg = jax.lax.stop_gradient
    prior = gaussian_divergence(pm, pl, sg(qm), sg(ql), kind)
    posterior = gaussian_divergence(sg(pm), sg(pl), qm, ql, kind)
    weighted = 2 * alpha * prior + 2 * (1 - alpha) * posterior
    # Preserve raw divergence metrics and the existing one-bit free region.
    return sg(prior) + (weighted - sg(weighted))


def free_bit(kl):
    """max(1, KL_bits) for the entire multivariate event."""
    return jnp.maximum(1.0, kl / jnp.log(2.0))


def make_input(embedding, previous_action, previous_reward, done):
    # At reset there is no previous action: use an all-zero one-hot sentinel.
    action = jax.nn.one_hot(previous_action, 5) * (~done)[..., None]
    reward = jnp.where(done, 0.0, previous_reward)[..., None]
    return jnp.concatenate((action, reward, embedding), axis=-1)


class LRUCell(eqx.Module):
    """Stable diagonal complex LRU, with real-valued trainable parameters."""

    nu: jax.Array
    theta: jax.Array
    gamma: jax.Array
    b_real: jax.Array
    b_imag: jax.Array
    c_real: jax.Array
    c_imag: jax.Array
    skip: jax.Array

    def __init__(self, input_size, hidden_size, key):
        keys = jax.random.split(key, 7)
        radius = jax.random.uniform(keys[0], (hidden_size,), minval=0.0, maxval=0.99)
        self.nu = jnp.log(-jnp.log(jnp.maximum(radius, 1e-4)))
        self.theta = jnp.log(
            jax.random.uniform(keys[1], (hidden_size,), minval=1e-4, maxval=2 * jnp.pi)
        )
        self.gamma = jnp.log(jnp.sqrt(1 - radius**2))
        self.b_real = jax.random.normal(keys[2], (hidden_size, input_size)) / jnp.sqrt(
            2 * input_size
        )
        self.b_imag = jax.random.normal(keys[3], (hidden_size, input_size)) / jnp.sqrt(
            2 * input_size
        )
        self.c_real = jax.random.normal(keys[4], (hidden_size, hidden_size)) / jnp.sqrt(
            hidden_size
        )
        self.c_imag = jax.random.normal(keys[5], (hidden_size, hidden_size)) / jnp.sqrt(
            hidden_size
        )
        self.skip = jax.random.normal(keys[6], (hidden_size, input_size)) / jnp.sqrt(
            input_size
        )

    def __call__(self, state, x, mask):
        eigenvalue = jnp.exp(-jnp.exp(self.nu) + 1j * jnp.exp(self.theta))
        b = jnp.exp(self.gamma)[:, None] * (self.b_real + 1j * self.b_imag)
        state = eigenvalue * (state * mask) + b @ x
        output = jnp.real((self.c_real + 1j * self.c_imag) @ state) + self.skip @ x
        return state, jax.nn.leaky_relu(output)


class Branch(eqx.Module):
    cnn: nn.Sequential
    cells: tuple
    prediction: nn.Linear | None
    memory_projection: nn.Linear | None
    prior: nn.Linear | None
    posterior: nn.Linear | None
    head: nn.Sequential
    variant: str = eqx.field(static=True)
    hidden: int = eqx.field(static=True)
    layers: int = eqx.field(static=True)
    latent_dim: int = eqx.field(static=True)
    dropout: float = eqx.field(static=True)
    posterior_input: str = eqx.field(static=True)
    residual: bool = eqx.field(static=True)

    def __init__(
        self,
        cnn,
        variant,
        output_size,
        hidden,
        layers,
        dropout,
        key,
        posterior_input=None,
        residual=None,
    ):
        posterior_input, residual = resolve_memory_options(
            variant, posterior_input, residual
        )
        self.posterior_input = posterior_input
        self.residual = residual
        keys = jax.random.split(key, layers + 5)
        self.cnn, self.variant = cnn, variant
        self.hidden, self.layers, self.dropout = hidden, layers, dropout
        # A complex coordinate comprises two real scalar state dimensions.
        self.latent_dim = (2 * hidden * layers * 2) // 3
        self.cells = (
            tuple(
                LRUCell(518 if i == 0 else hidden, hidden, keys[i])
                for i in range(layers)
            )
            if variant != "mlp"
            else ()
        )
        is_ib = variant in IB_DIVERGENCES
        self.memory_projection = (
            nn.Linear(hidden, 256, key=keys[-5])
            if variant != "mlp" and not is_ib
            else None
        )
        self.prediction = (
            nn.Linear(
                518 + (2 * self.latent_dim if is_ib else hidden),
                5 * 513,
                key=jax.random.fold_in(key, 987),
            )
            if variant.endswith("_prediction")
            else None
        )
        self.prior = (
            nn.Linear(518, 2 * self.latent_dim, key=keys[-4]) if is_ib else None
        )
        self.posterior = (
            nn.Linear(
                518 + (hidden if posterior_input == "output" else 2 * hidden * layers),
                2 * self.latent_dim,
                key=keys[-3],
            )
            if is_ib
            else None
        )
        head_size = 518 + (
            2 * self.latent_dim if is_ib else 256 if variant != "mlp" else 0
        )
        self.head = nn.Sequential(
            [
                nn.Linear(head_size, 256, key=keys[-2]),
                nn.LayerNorm(256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(256, output_size, key=keys[-1]),
            ]
        )

    def encode(self, obs, action, reward, done):
        # The upstream renderer returns uint8 RGB observations.
        embedding = jax.vmap(self.cnn)(
            obs.transpose(0, 3, 1, 2).astype(jnp.float32) / 255.0
        )
        return make_input(embedding.reshape(obs.shape[0], -1), action, reward, done)

    def step(self, state, x, done, key, training):
        state = jnp.where(done, jnp.zeros_like(state), state)
        if self.variant == "mlp":
            return state, self.head(x), jnp.array(0.0)
        masks = jnp.ones(state.shape, dtype=jnp.float32)
        if self.variant == "state_dropout" and training:
            # Drop the carried state, sharing each mask across its real/imag parts.
            # No inverted scaling: retained diagonal dynamics remain contractive.
            masks = jax.random.bernoulli(key, 1 - self.dropout, state.shape)
        states, y = [], x
        for i, cell in enumerate(self.cells):
            previous = y
            s, y = cell(state[i], y, masks[i])
            if self.residual and i > 0:
                y = y + previous
            states.append(s)
        state = jnp.stack(states)
        kl = jnp.array(0.0)
        if self.variant in IB_DIVERGENCES:
            flat_state = jnp.concatenate((state.real.ravel(), state.imag.ravel()))
            pm, pl = jnp.split(self.prior(x), 2)
            qm, ql = jnp.split(
                self.posterior(
                    jnp.concatenate(
                        (x, y if self.posterior_input == "output" else flat_state)
                    )
                ),
                2,
            )
            pl, ql = jnp.clip(pl, -10, 10), jnp.clip(ql, -10, 10)
            kl = balanced_divergence(
                pm,
                pl,
                qm,
                ql,
                IB_DIVERGENCES[self.variant],
                KL_BALANCES.get(self.variant, 0.5),
            )
            features = jnp.concatenate((x, qm, ql))
        else:
            features = jnp.concatenate((x, self.memory_projection(y)))
        if self.prediction is not None:
            kl = (
                kl,
                self.prediction(
                    features
                    if self.variant in IB_DIVERGENCES
                    else jnp.concatenate((x, y))
                ).reshape(5, 513),
            )
        return state, self.head(features), kl

    def parallel_sequence(self, initial_state, inputs, done, keys, training):
        """Associative affine scan: same recurrence and reset/dropout masks as step."""
        masks = jnp.ones((inputs.shape[0], self.layers, self.hidden))
        if self.variant == "state_dropout" and training:
            masks = jax.vmap(
                lambda key: jax.random.bernoulli(
                    key, 1 - self.dropout, (self.layers, self.hidden)
                )
            )(keys)

        def compose(left, right):
            a1, b1 = left
            a2, b2 = right
            return a2 * a1, a2 * b1 + b2

        layer_states, y = [], inputs
        for i, cell in enumerate(self.cells):
            previous = y
            eigenvalue = jnp.exp(-jnp.exp(cell.nu) + 1j * jnp.exp(cell.theta))
            factors = jnp.where(done[:, None], 0.0, eigenvalue * masks[:, i])
            b = jnp.exp(cell.gamma)[:, None] * (cell.b_real + 1j * cell.b_imag)
            drives = y.astype(jnp.complex64) @ b.T
            factors, offsets = jax.lax.associative_scan(compose, (factors, drives))
            states = factors * initial_state[i] + offsets
            c = cell.c_real + 1j * cell.c_imag
            y = jax.nn.leaky_relu(jnp.real(states @ c.T) + y @ cell.skip.T)
            if self.residual and i > 0:
                y = y + previous
            layer_states.append(states)
        states = jnp.stack(layer_states, axis=1)
        kl = jnp.zeros(inputs.shape[0])
        if self.variant in IB_DIVERGENCES:
            flat_state = jnp.concatenate(
                (
                    states.real.reshape(inputs.shape[0], -1),
                    states.imag.reshape(inputs.shape[0], -1),
                ),
                axis=-1,
            )
            pm, pl = jnp.split(jax.vmap(self.prior)(inputs), 2, axis=-1)
            qm, ql = jnp.split(
                jax.vmap(self.posterior)(
                    jnp.concatenate(
                        (inputs, y if self.posterior_input == "output" else flat_state),
                        axis=-1,
                    )
                ),
                2,
                axis=-1,
            )
            pl, ql = jnp.clip(pl, -10, 10), jnp.clip(ql, -10, 10)
            kl = balanced_divergence(
                pm,
                pl,
                qm,
                ql,
                IB_DIVERGENCES[self.variant],
                KL_BALANCES.get(self.variant, 0.5),
            )
            features = jnp.concatenate((inputs, qm, ql), axis=-1)
        else:
            features = jnp.concatenate(
                (inputs, jax.vmap(self.memory_projection)(y)), axis=-1
            )
        if self.prediction is not None:
            kl = (
                kl,
                jax.vmap(self.prediction)(
                    features
                    if self.variant in IB_DIVERGENCES
                    else jnp.concatenate((inputs, y), axis=-1)
                ).reshape(inputs.shape[0], 5, 513),
            )
        return states[-1], jax.vmap(self.head)(features), kl


class MemoryActorCritic(eqx.Module):
    actor: Branch
    critic: Branch

    def __init__(
        self,
        variant,
        key,
        *,
        hidden=512,
        layers=2,
        dropout=0.1,
        obs_size=128,
        posterior_input=None,
        residual=None,
    ):
        cnn_key, actor_key, critic_key = jax.random.split(key, 3)
        upstream = ActorCritic(cnn_key, obs_size)
        self.actor = Branch(
            upstream.actor_cnn,
            variant,
            5,
            hidden,
            layers,
            dropout,
            actor_key,
            posterior_input,
            residual,
        )
        self.critic = Branch(
            upstream.critic_cnn,
            variant,
            1,
            hidden,
            layers,
            dropout,
            critic_key,
            posterior_input,
            residual,
        )

    def initial_state(self, batch_size):
        shape = (batch_size, self.actor.layers, self.actor.hidden)
        return jnp.zeros(shape, jnp.complex64), jnp.zeros(shape, jnp.complex64)

    def encode(self, obs, action, reward, done):
        return (
            self.actor.encode(obs, action, reward, done),
            self.critic.encode(obs, action, reward, done),
        )

    def step_encoded(self, state, encoded, done, keys, training):
        actor_keys, critic_keys = jax.vmap(
            lambda k: jax.random.split(k, 2), out_axes=1
        )(keys)
        a, logits, akl = jax.vmap(
            lambda s, x, d, k: self.actor.step(s, x, d, k, training)
        )(state[0], encoded[0], done, actor_keys)
        c, value, ckl = jax.vmap(
            lambda s, x, d, k: self.critic.step(s, x, d, k, training)
        )(state[1], encoded[1], done, critic_keys)
        return (a, c), logits, value[..., 0], (akl, ckl)

    def step(self, state, obs, action, reward, done, keys, training):
        return self.step_encoded(
            state, self.encode(obs, action, reward, done), done, keys, training
        )

    def sequence(
        self,
        state,
        obs,
        action,
        reward,
        done,
        keys,
        training=True,
        detach_encoder=False,
    ):
        t, b = obs.shape[:2]
        encoded = self.encode(
            obs.reshape((-1,) + obs.shape[2:]),
            action.ravel(),
            reward.ravel(),
            done.ravel(),
        )
        if detach_encoder:
            encoded = jax.tree.map(jax.lax.stop_gradient, encoded)
        encoded = tuple(x.reshape(t, b, -1) for x in encoded)

        if self.actor.variant != "mlp":
            branch_keys = jax.vmap(jax.vmap(lambda key: jax.random.split(key, 2)))(keys)
            a, logits, akl = jax.vmap(
                lambda s, x, d, k: self.actor.parallel_sequence(s, x, d, k, training),
                in_axes=(0, 1, 1, 1),
                out_axes=(0, 1, 1),
            )(state[0], encoded[0], done, branch_keys[:, :, 0])
            c, value, ckl = jax.vmap(
                lambda s, x, d, k: self.critic.parallel_sequence(s, x, d, k, training),
                in_axes=(0, 1, 1, 1),
                out_axes=(0, 1, 1),
            )(state[1], encoded[1], done, branch_keys[:, :, 1])
            return (a, c), (logits, value[..., 0], (akl, ckl))

        def body(carry, xs):
            enc, d, k = xs
            carry, logits, value, kl = self.step_encoded(carry, enc, d, k, training)
            return carry, (logits, value, kl)

        return jax.lax.scan(body, state, (encoded, done, keys))


def normalized_excess(divergence_nats, scale_bits):
    """Fixed RMS scaling after the raw full-event one-bit free region."""
    return jnp.maximum(
        divergence_nats / jnp.log(2.0) - 1.0, 0.0
    ) / jax.lax.stop_gradient(scale_bits)


def prediction_losses(predictions, action, next_embedding, reward, next_done):
    """Selected-action MSE; reset images are excluded, terminal rewards retained."""
    chosen = jnp.take_along_axis(predictions, action[..., None, None], axis=-2)[
        ..., 0, :
    ]
    error = jnp.mean(
        (chosen[..., :512] - jax.lax.stop_gradient(next_embedding)) ** 2, -1
    )
    valid = ~next_done
    observation_loss = jnp.where(valid, error, 0.0).sum() / jnp.maximum(valid.sum(), 1)
    reward_loss = jnp.mean((chosen[..., 512] - reward) ** 2)
    return observation_loss, reward_loss
