"""Small, explicit PPO primitives used by JAX-based continuous-control trainers."""
from typing import TypedDict, cast

import jax
import jax.numpy as jnp


class Layer(TypedDict):
    weight: jax.Array
    bias: jax.Array


class PolicyParams(TypedDict):
    l1: Layer
    l2: Layer
    actor: Layer
    critic: Layer
    log_std: jax.Array


class AdamState(TypedDict):
    count: jax.Array
    m: PolicyParams
    v: PolicyParams


class Batch(TypedDict):
    observations: jax.Array
    actions: jax.Array
    old_logp: jax.Array
    advantages: jax.Array
    returns: jax.Array


def init_policy(key: jax.Array, observation_size: int, action_size: int, hidden: int = 64) -> PolicyParams:
    keys = jax.random.split(key, 4)
    def layer(key: jax.Array, n_in: int, n_out: int, scale: float = 1.) -> Layer:
        return {'weight': jax.random.normal(key, (n_in, n_out)) * (scale / n_in ** .5),
                'bias': jnp.zeros(n_out)}
    return {'l1': layer(keys[0], observation_size, hidden),
            'l2': layer(keys[1], hidden, hidden),
            'actor': layer(keys[2], hidden, action_size, .01),
            'critic': layer(keys[3], hidden, 1),
            'log_std': jnp.full((action_size,), -.7)}


def policy(params: PolicyParams, observations: jax.Array) -> tuple[jax.Array, jax.Array]:
    def linear(x: jax.Array, layer: Layer) -> jax.Array:
        return x @ layer['weight'] + layer['bias']
    hidden = jnp.tanh(linear(observations, params['l1']))
    hidden = jnp.tanh(linear(hidden, params['l2']))
    return linear(hidden, params['actor']), linear(hidden, params['critic']).squeeze(-1)


def log_probability(params: PolicyParams, mean: jax.Array, actions: jax.Array) -> jax.Array:
    log_std = jnp.clip(params['log_std'], -5., 2.)
    return (-.5 * (((actions - mean) / jnp.exp(log_std)) ** 2 + 2 * log_std + jnp.log(2 * jnp.pi))).sum(axis=-1)


def generalized_advantage(
    rewards: jax.Array, values: jax.Array, dones: jax.Array, bootstrap_value: jax.Array,
    gamma: float = .99, lam: float = .95,
) -> jax.Array:
    def step(
        carry: tuple[jax.Array, jax.Array], sample: tuple[jax.Array, jax.Array, jax.Array],
    ) -> tuple[tuple[jax.Array, jax.Array], jax.Array]:
        advantage, next_value = carry
        reward, value, done = sample
        live = 1. - done
        delta = reward + gamma * live * next_value - value
        advantage = delta + gamma * lam * live * advantage
        return (advantage, value), advantage
    _, advantages = jax.lax.scan(step, (jnp.zeros_like(bootstrap_value), bootstrap_value),
                                 (rewards, values, dones), reverse=True)
    return advantages


def adam_init(params: PolicyParams) -> AdamState:
    return {'count': jnp.array(0), 'm': cast(PolicyParams, jax.tree.map(jnp.zeros_like, params)),
            'v': cast(PolicyParams, jax.tree.map(jnp.zeros_like, params))}


def loss(params: PolicyParams, batch: Batch) -> tuple[jax.Array, dict[str, jax.Array]]:
    mean, values = policy(params, batch['observations'])
    logp = log_probability(params, mean, batch['actions'])
    ratio = jnp.exp(logp - batch['old_logp'])
    advantages = batch['advantages']
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    actor_loss = -jnp.minimum(ratio * advantages, jnp.clip(ratio, .8, 1.2) * advantages).mean()
    value_loss = .5 * jnp.square(values - batch['returns']).mean()
    entropy = (jnp.clip(params['log_std'], -5., 2.) + .5 * (1. + jnp.log(2 * jnp.pi))).sum()
    total = actor_loss + .5 * value_loss - .001 * entropy
    return total, {'loss': total, 'actor_loss': actor_loss, 'value_loss': value_loss, 'entropy': entropy}


def update(
    params: PolicyParams, state: AdamState, batch: Batch, learning_rate: float,
) -> tuple[PolicyParams, AdamState, dict[str, jax.Array]]:
    # JAX's pytree helpers preserve the policy layout while replacing its array leaves.
    (_, metrics), gradients = cast(
        tuple[tuple[jax.Array, dict[str, jax.Array]], PolicyParams],
        jax.value_and_grad(loss, has_aux=True)(params, batch),  # pyright: ignore[reportUnknownMemberType]
    )
    leaves = cast(list[jax.Array], jax.tree_util.tree_leaves(gradients))
    norm = jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))

    def clip_gradient(x: jax.Array) -> jax.Array:
        return x * jnp.minimum(1., .5 / (norm + 1e-8))

    gradients = cast(PolicyParams, jax.tree.map(clip_gradient, gradients))
    count = state['count'] + 1

    def first_moment(a: jax.Array, b: jax.Array) -> jax.Array:
        return .9 * a + .1 * b

    def second_moment(a: jax.Array, b: jax.Array) -> jax.Array:
        return .999 * a + .001 * b * b

    def apply_update(p: jax.Array, a: jax.Array, b: jax.Array) -> jax.Array:
        return p - learning_rate * (a / (1. - .9 ** count)) / (jnp.sqrt(b / (1. - .999 ** count)) + 1e-8)

    m = cast(PolicyParams, jax.tree.map(first_moment, state['m'], gradients))
    v = cast(PolicyParams, jax.tree.map(second_moment, state['v'], gradients))
    params = cast(PolicyParams, jax.tree.map(apply_update, params, m, v))
    return params, {'count': count, 'm': m, 'v': v}, {**metrics, 'gradient_norm': norm}
