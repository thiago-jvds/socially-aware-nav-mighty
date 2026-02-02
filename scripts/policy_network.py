"""
Phase 3: Neural Network Policy for MIGHTY Evasion

This module provides a neural network that maps observations of the ego agent
and adversary to MIGHTY cost weights, enabling learned trajectory optimization.

Key function:
    EvasionPolicy: Flax neural network module that outputs cost weights theta
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import flax.linen as nn
from typing import Dict


class EvasionPolicy(nn.Module):
    """
    Neural network policy that maps observations to MIGHTY cost weights.

    The policy observes the current state (ego and adversary positions/velocities)
    and outputs 7 cost weights that parameterize the MIGHTY trajectory optimizer.

    Architecture:
        - Input: 12D observation vector
        - Hidden: 2 layers of 64 units with ReLU activation
        - Output: 7 cost weights (positive via softplus + base values)

    Observation format:
        [ego_pos_x, ego_pos_y, ego_pos_z,      # 3D position of ego agent
         ego_vel_x, ego_vel_y, ego_vel_z,      # 3D velocity of ego agent
         adv_pos_x, adv_pos_y, adv_pos_z,      # 3D position of adversary
         adv_vel_x, adv_vel_y, adv_vel_z]      # 3D velocity of adversary

    Output cost weights:
        - time_weight: Penalty on trajectory duration
        - dyn_weight: Dynamic obstacle avoidance weight
        - stat_weight: Static obstacle avoidance weight
        - accel_weight: Acceleration minimization weight
        - jerk_weight: Jerk (smoothness) minimization weight
        - vel_constr_weight: Velocity constraint violation penalty
        - acc_constr_weight: Acceleration constraint violation penalty
    """

    hidden_dim: int = 64  # Number of units in hidden layers

    @nn.compact
    def __call__(self, obs):
        """
        Forward pass: observation -> cost weights.

        Args:
            obs: (12,) observation vector [ego_pos, ego_vel, adv_pos, adv_vel]

        Returns:
            theta: dict of 7 cost weights, all positive scalars
        """
        # Hidden layer 1
        x = nn.Dense(self.hidden_dim, name='hidden1')(obs)
        x = nn.relu(x)

        # Hidden layer 2
        x = nn.Dense(self.hidden_dim, name='hidden2')(x)
        x = nn.relu(x)

        # Output layer: 7 raw weight values
        raw_weights = nn.Dense(7, name='output')(x)

        # Base values (defaults from manual tuning)
        # These are added to ensure weights stay in a reasonable range
        base = jnp.array([
            10.0,  # time_weight
            0.1,   # dyn_weight
            0.1,   # stat_weight
            0.1,   # accel_weight
            0.1,   # jerk_weight
            10.0,  # vel_constr_weight
            1.0,   # acc_constr_weight
        ])

        # Apply softplus to ensure positivity, then add base
        # softplus(x) = log(1 + exp(x)) is always positive and smooth
        weights = jax.nn.softplus(raw_weights) + base

        # Return as dict for easy integration with MIGHTY solver
        return {
            'time_weight': weights[0],
            'dyn_weight': weights[1],
            'stat_weight': weights[2],
            'accel_weight': weights[3],
            'jerk_weight': weights[4],
            'vel_constr_weight': weights[5],
            'acc_constr_weight': weights[6],
        }


class EvasionPolicyWithGoal(nn.Module):
    """
    Extended policy that also outputs goal position offset.

    In addition to cost weights, this policy can adjust the goal position
    to enable more dynamic evasion behavior.

    Output:
        - theta: dict of 7 cost weights (same as EvasionPolicy)
        - goal_offset: (3,) offset to apply to goal position (bounded by tanh)
    """

    hidden_dim: int = 64
    max_goal_offset: float = 0.5  # Maximum goal offset in meters

    @nn.compact
    def __call__(self, obs):
        """
        Forward pass: observation -> cost weights + goal offset.

        Args:
            obs: (12,) observation vector

        Returns:
            theta: dict of 7 cost weights + 'goal_offset' key
        """
        # Shared hidden layers
        x = nn.Dense(self.hidden_dim, name='hidden1')(obs)
        x = nn.relu(x)

        x = nn.Dense(self.hidden_dim, name='hidden2')(x)
        x = nn.relu(x)

        # Head 1: Cost weights (same as base policy)
        raw_weights = nn.Dense(7, name='weights_head')(x)
        base = jnp.array([10.0, 0.1, 0.1, 0.1, 0.1, 10.0, 1.0])
        weights = jax.nn.softplus(raw_weights) + base

        # Head 2: Goal offset (bounded by tanh)
        raw_goal_offset = nn.Dense(3, name='goal_head')(x)
        goal_offset = self.max_goal_offset * jnp.tanh(raw_goal_offset)

        return {
            'time_weight': weights[0],
            'dyn_weight': weights[1],
            'stat_weight': weights[2],
            'accel_weight': weights[3],
            'jerk_weight': weights[4],
            'vel_constr_weight': weights[5],
            'acc_constr_weight': weights[6],
            'goal_offset': goal_offset,  # (3,) offset to add to xf
        }


def create_observation(ego_pos, ego_vel, adv_pos, adv_vel):
    """
    Create observation vector from state components.

    Args:
        ego_pos: (3,) ego agent position
        ego_vel: (3,) ego agent velocity
        adv_pos: (3,) adversary position
        adv_vel: (3,) adversary velocity

    Returns:
        obs: (12,) observation vector
    """
    return jnp.concatenate([ego_pos, ego_vel, adv_pos, adv_vel])


def create_relative_observation(ego_pos, ego_vel, adv_pos, adv_vel):
    """
    Create observation vector with relative coordinates.

    Using relative observations can help the policy generalize better
    across different spatial configurations.

    Args:
        ego_pos: (3,) ego agent position
        ego_vel: (3,) ego agent velocity
        adv_pos: (3,) adversary position
        adv_vel: (3,) adversary velocity

    Returns:
        obs: (12,) observation vector with relative features
    """
    relative_pos = adv_pos - ego_pos
    relative_vel = adv_vel - ego_vel

    return jnp.concatenate([
        ego_pos,        # Absolute ego position (for spatial awareness)
        ego_vel,        # Absolute ego velocity
        relative_pos,   # Relative adversary position
        relative_vel,   # Relative adversary velocity
    ])


def initialize_policy(policy_class=EvasionPolicy, hidden_dim=64, seed=0):
    """
    Initialize policy network parameters.

    Args:
        policy_class: Policy class to instantiate (EvasionPolicy or EvasionPolicyWithGoal)
        hidden_dim: Number of hidden units
        seed: Random seed for initialization

    Returns:
        policy: Initialized policy module
        params: Initial network parameters
        apply_fn: Function to apply policy (params, obs) -> theta
    """
    # Create policy instance
    policy = policy_class(hidden_dim=hidden_dim)

    # Initialize with dummy observation
    rng = jax.random.PRNGKey(seed)
    dummy_obs = jnp.zeros(12)

    # Initialize parameters
    params = policy.init(rng, dummy_obs)

    # Create apply function
    def apply_fn(params, obs):
        return policy.apply(params, obs)

    return policy, params, apply_fn


def count_parameters(params):
    """
    Count total number of trainable parameters.

    Args:
        params: Flax parameter pytree

    Returns:
        total_params: Total number of parameters
    """
    return sum(x.size for x in jax.tree_util.tree_leaves(params))


def policy_forward(params, policy, obs):
    """
    Convenience function for forward pass.

    Args:
        params: Network parameters
        policy: Policy module instance
        obs: (12,) or (batch, 12) observation(s)

    Returns:
        theta: Cost weights dict (or batch of dicts)
    """
    return policy.apply(params, obs)


# Example usage
if __name__ == "__main__":
    print("=" * 70)
    print("EVASION POLICY - Example Usage")
    print("=" * 70)

    # Initialize policy
    policy, params, apply_fn = initialize_policy(EvasionPolicy, hidden_dim=64, seed=42)

    print(f"\nPolicy architecture:")
    print(f"  Input: 12D observation")
    print(f"  Hidden: 2 layers × 64 units (ReLU)")
    print(f"  Output: 7 cost weights")
    print(f"  Total parameters: {count_parameters(params)}")

    # Create dummy observation
    ego_pos = jnp.array([0.0, 0.0, 1.0])
    ego_vel = jnp.array([1.0, 0.5, 0.0])
    adv_pos = jnp.array([5.0, 2.0, 1.0])
    adv_vel = jnp.array([-0.5, 0.3, 0.0])

    obs = create_observation(ego_pos, ego_vel, adv_pos, adv_vel)
    print(f"\nObservation shape: {obs.shape}")

    # Forward pass
    theta = apply_fn(params, obs)

    print(f"\nOutput cost weights:")
    for key, val in theta.items():
        print(f"  {key:20s}: {float(val):.4f}")

    # Test with relative observation
    obs_rel = create_relative_observation(ego_pos, ego_vel, adv_pos, adv_vel)
    theta_rel = apply_fn(params, obs_rel)

    print(f"\nWith relative observation:")
    for key, val in theta_rel.items():
        print(f"  {key:20s}: {float(val):.4f}")

    # Test extended policy with goal offset
    print("\n" + "=" * 70)
    print("EXTENDED POLICY - With Goal Offset")
    print("=" * 70)

    policy_ext, params_ext, apply_fn_ext = initialize_policy(
        EvasionPolicyWithGoal, hidden_dim=64, seed=42
    )

    print(f"\nTotal parameters: {count_parameters(params_ext)}")

    theta_ext = apply_fn_ext(params_ext, obs)
    print(f"\nOutput:")
    for key, val in theta_ext.items():
        if key == 'goal_offset':
            print(f"  {key:20s}: {val}")
        else:
            print(f"  {key:20s}: {float(val):.4f}")

    print("\n✓ Policy network ready for training!")
