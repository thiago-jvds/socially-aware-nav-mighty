"""
Phase 2: Differentiable MIGHTY Trajectory Optimizer using JAXopt

This module provides a JAX-native optimizer wrapper that enables computing
gradients of the optimal trajectory w.r.t. cost weights using implicit differentiation.

Key function:
    solve_mighty_jax(theta, env_params, z_init) -> z_star

    Returns optimal trajectory z* that is differentiable w.r.t. theta via
    implicit function theorem (no unrolling, no Hessian storage).
"""

import jax
import jax.numpy as jnp
from jax import config
config.update("jax_enable_x64", True)

import jaxopt
from mighty_jax import evaluate_objective_jax, reconstruct_jax, sample_trajectory_jax


def solve_mighty_jax(theta, env_params, z_init, maxiter=50, tol=1e-5, verbose=False):
    """
    Solve MIGHTY trajectory optimization using JAXopt L-BFGS.

    The returned optimal trajectory z* is differentiable w.r.t. theta using
    implicit differentiation (Blondel et al., NeurIPS 2022).

    Args:
        theta: dict of cost weights from neural network
            Keys: 'time_weight', 'dyn_weight', 'stat_weight', 'accel_weight',
                  'jerk_weight', 'vel_constr_weight', 'acc_constr_weight'
        env_params: dict of environment parameters
            Keys: 'x0', 'v0', 'a0', 'xf', 'vf', 'af', 'wps', 'free_cps_indices',
                  'obstacles', 'A_stat', 'b_stat', 'V_max', 'A_max', 'N_samp',
                  'N_per_seg', 'Co', 'Cw', 'c_ellipsoid'
        z_init: (K,) initial guess for decision vector
        maxiter: maximum L-BFGS iterations (default: 50, matches C++ MIGHTY)
        tol: convergence tolerance (default: 1e-5)
        verbose: if True, print optimization progress

    Returns:
        z_star: (K,) optimal decision vector, differentiable w.r.t. theta

    Example:
        >>> theta = {'time_weight': 10.0, 'dyn_weight': 0.1, ...}
        >>> z_star = solve_mighty_jax(theta, env_params, z_init)
        >>>
        >>> # Compute gradient of z* w.r.t. theta
        >>> def get_z_star(theta):
        >>>     return solve_mighty_jax(theta, env_params, z_init)
        >>> dz_dtheta = jax.jacobian(get_z_star)(theta)
    """
    def cost_fn(z):
        """Cost function for optimizer (theta is captured from closure)."""
        return evaluate_objective_jax(z, theta, env_params)

    # Create L-BFGS solver with implicit differentiation
    solver = jaxopt.LBFGS(
        fun=cost_fn,
        maxiter=maxiter,
        tol=tol,
        implicit_diff=True,  # KEY: enables dz*/dtheta via implicit function theorem
        jit=True,            # JIT compile for performance
        unroll=False,        # Don't unroll (use implicit diff instead)
    )

    # Solve optimization
    result = solver.run(z_init)

    if verbose:
        print(f"Optimization converged: {result.state.error < tol}")
        print(f"Final cost: {result.state.value:.6f}")
        print(f"Gradient norm: {result.state.error:.6e}")
        print(f"Iterations: {result.state.iter_num}")

    # Return optimal z* (differentiable w.r.t. theta)
    return result.params


def solve_mighty_with_info(theta, env_params, z_init, maxiter=50, tol=1e-5):
    """
    Solve MIGHTY optimization and return both z* and optimization info.

    Similar to solve_mighty_jax but also returns the full optimization result
    for debugging and analysis.

    Returns:
        z_star: (K,) optimal decision vector
        info: dict with optimization diagnostics
            - 'converged': bool, whether optimization converged
            - 'cost': final cost value
            - 'grad_norm': gradient norm at solution
            - 'iterations': number of iterations taken
            - 'state': full JAXopt state object
    """
    def cost_fn(z):
        return evaluate_objective_jax(z, theta, env_params)

    solver = jaxopt.LBFGS(
        fun=cost_fn,
        maxiter=maxiter,
        tol=tol,
        implicit_diff=True,
        jit=True,
        unroll=False,
    )

    result = solver.run(z_init)

    info = {
        'converged': result.state.error < tol,
        'cost': float(result.state.value),
        'grad_norm': float(result.state.error),
        'iterations': int(result.state.iter_num),
        'state': result.state,
    }

    return result.params, info


def solve_and_extract_trajectory(theta, env_params, z_init, N_samp=None, maxiter=50, tol=1e-5):
    """
    Solve MIGHTY optimization and extract the resulting trajectory.

    Convenience function that combines solve_mighty_jax with trajectory extraction.
    Useful for end-to-end pipelines where you need the actual trajectory points.

    Args:
        theta: cost weights dict
        env_params: environment parameters dict
        z_init: initial guess
        N_samp: number of trajectory samples (if None, uses env_params['N_samp'])
        maxiter: max L-BFGS iterations
        tol: convergence tolerance

    Returns:
        z_star: (K,) optimal decision vector
        trajectory: dict with keys:
            - 'positions': (N_total, 3) trajectory positions
            - 'velocities': (N_total, 3) trajectory velocities
            - 'accelerations': (N_total, 3) trajectory accelerations
            - 'times': (N_total,) time stamps
            - 'CP': (M, 6, 3) Bézier control points
            - 'T': (M,) segment durations
    """
    # Solve optimization
    z_star = solve_mighty_jax(theta, env_params, z_init, maxiter=maxiter, tol=tol)

    # Extract trajectory
    CP, T = reconstruct_jax(
        z_star,
        env_params['x0'], env_params['v0'], env_params['a0'],
        env_params['xf'], env_params['vf'], env_params['af'],
        env_params['wps'],
        env_params['free_cps_indices']
    )

    # Sample trajectory
    if N_samp is None:
        N_per_seg = env_params.get('N_per_seg', 3)
    else:
        M = CP.shape[0]
        N_per_seg = max(1, N_samp // M)

    positions, velocities, accelerations, times = sample_trajectory_jax(CP, T, N_per_seg)

    trajectory = {
        'positions': positions,
        'velocities': velocities,
        'accelerations': accelerations,
        'times': times,
        'CP': CP,
        'T': T,
    }

    return z_star, trajectory


def compute_trajectory_gradient_wrt_theta(theta, env_params, z_init, maxiter=50, tol=1e-5):
    """
    Compute gradient of trajectory positions w.r.t. cost weights theta.

    This demonstrates the end-to-end differentiability:
        theta -> solve_mighty_jax -> z* -> trajectory -> positions

    Useful for analyzing sensitivity of trajectory to different cost weights.

    Args:
        theta: cost weights dict
        env_params: environment parameters dict
        z_init: initial guess
        maxiter: max L-BFGS iterations
        tol: convergence tolerance

    Returns:
        grad_dict: dict mapping each theta key to gradient array
            - grad_dict['time_weight']: (N_total, 3) gradient of positions w.r.t. time_weight
            - etc. for all 7 cost weights
    """
    def get_trajectory_positions(theta_dict):
        """Extract trajectory positions as function of theta."""
        z_star, traj = solve_and_extract_trajectory(
            theta_dict, env_params, z_init, maxiter=maxiter, tol=tol
        )
        return traj['positions']

    # Compute Jacobian: d(positions)/d(theta)
    # This will be a dict of arrays, one for each theta key
    jacobian_fn = jax.jacrev(get_trajectory_positions)
    grad_dict = jacobian_fn(theta)

    return grad_dict


# Alternative solver: Unrolled Gradient Descent
def solve_mighty_unrolled(theta, env_params, z_init, num_steps=20, lr=0.01):
    """
    Solve MIGHTY using unrolled gradient descent.

    This is an alternative to solve_mighty_jax that uses explicit unrolling
    instead of implicit differentiation. Simpler but uses more memory
    (stores all intermediate states for backprop).

    Use this for:
    - Debugging / comparison with implicit diff version
    - Cases where implicit differentiation has issues
    - Very small problems where memory isn't a concern

    Args:
        theta: cost weights dict
        env_params: environment parameters dict
        z_init: initial guess
        num_steps: number of gradient descent steps (default: 20)
        lr: learning rate (default: 0.01)

    Returns:
        z: (K,) trajectory after num_steps, differentiable w.r.t. theta

    Note:
        This doesn't necessarily converge to optimal z*, but provides a
        differentiable approximation. Increase num_steps for better convergence.
    """
    z = z_init

    for _ in range(num_steps):
        # Compute gradient of cost w.r.t. z
        grad = jax.grad(evaluate_objective_jax, argnums=0)(z, theta, env_params)

        # Gradient descent step
        z = z - lr * grad

    return z


def solve_mighty_unrolled_with_linesearch(theta, env_params, z_init, num_steps=20, base_lr=0.1):
    """
    Unrolled gradient descent with simple backtracking line search.

    More robust than fixed learning rate, but still uses unrolled autodiff.

    Args:
        theta: cost weights dict
        env_params: environment parameters dict
        z_init: initial guess
        num_steps: number of gradient descent steps
        base_lr: initial learning rate for line search

    Returns:
        z: trajectory after num_steps
    """
    z = z_init

    for _ in range(num_steps):
        grad = jax.grad(evaluate_objective_jax, argnums=0)(z, theta, env_params)

        # Simple backtracking line search
        lr = base_lr
        f0 = evaluate_objective_jax(z, theta, env_params)

        # Try decreasing step sizes until we get improvement
        for _ in range(5):
            z_new = z - lr * grad
            f_new = evaluate_objective_jax(z_new, theta, env_params)

            if f_new < f0:
                z = z_new
                break

            lr *= 0.5
        else:
            # No improvement found, take small step anyway
            z = z - 0.001 * grad

    return z
