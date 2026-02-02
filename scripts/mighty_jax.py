"""
JAX-differentiable MIGHTY trajectory optimizer cost function.

This module provides a fully differentiable implementation of the MIGHTY cost function
using JAX, enabling gradient-based training of neural network policies.
"""

import jax
import jax.numpy as jnp
from jax import config
from math import comb

# Enable 64-bit precision for numerical stability
config.update("jax_enable_x64", True)


# =============================================================================
# Utility Functions (from notebook Cell 10)
# =============================================================================

# Legacy Bernstein function (kept for backwards compatibility if needed)
def bernstein_basis_and_derivative_jax(n, tau):
    """
    Legacy function. Use bernstein5_val, bernstein4_val, bernstein3_val instead.
    """
    B = jnp.array([comb(n, j) * tau**j * (1 - tau)**(n - j) for j in range(n + 1)])
    if n == 0:
        dB = jnp.array([0.0])
    else:
        Bm1 = jnp.array([comb(n - 1, j) * tau**j * (1 - tau)**((n - 1) - j)
                        for j in range(n)])
        dB = jnp.zeros(n + 1)
        dB = dB.at[0].set(-n * Bm1[0])
        dB = dB.at[n].set(n * Bm1[-1])
        for j in range(1, n):
            dB = dB.at[j].add(n * (Bm1[j - 1] - Bm1[j]))
    return B, dB


def bernstein5_val(u, k):
    """Single value of degree-5 Bernstein basis function k at parameter u."""
    um = 1.0 - u
    coeffs = jnp.array([1.0, 5.0, 10.0, 10.0, 5.0, 1.0])  # binomial(5, k)
    return coeffs[k] * u**k * um**(5 - k)


def bernstein4_val(u, k):
    """Single value of degree-4 Bernstein basis function k at parameter u."""
    um = 1.0 - u
    coeffs = jnp.array([1.0, 4.0, 6.0, 4.0, 1.0])  # binomial(4, k)
    return coeffs[k] * u**k * um**(4 - k)


def bernstein3_val(u, k):
    """Single value of degree-3 Bernstein basis function k at parameter u."""
    um = 1.0 - u
    coeffs = jnp.array([1.0, 3.0, 3.0, 1.0])  # binomial(3, k)
    return coeffs[k] * u**k * um**(3 - k)


def reconstruct_jax(z, x0, v0, a0, xf, vf, af, wps, free_cps_indices):
    """
    Reconstruct full Bezier control points and segment times from decision vector.

    Args:
        z: Decision vector [free_CPs (F×3), σ_slacks (M)]
        x0, v0, a0: Start position, velocity, acceleration (3,)
        xf, vf, af: End position, velocity, acceleration (3,)
        wps: Global waypoints (M+1, 3)
        free_cps_indices: List of (seg, cp_idx) tuples for free control points

    Returns:
        CP: Control points (M, 6, 3)
        T: Segment durations (M,)
    """
    M = wps.shape[0] - 1
    F = len(free_cps_indices)

    # Unpack decision vector
    cp_free = z[:3 * F].reshape(F, 3)
    sig = z[3 * F:3 * F + M]

    # Segment times from exponential parameterization
    T = jnp.exp(sig)

    # Initialize control points
    CP = jnp.zeros((M, 6, 3))
    for idx_free, (seg, cp_idx) in enumerate(free_cps_indices):
        CP = CP.at[seg, cp_idx].set(cp_free[idx_free])

    # Inject start boundary conditions
    T0 = T[0]
    eps = 1e-10  # Numerical stability
    CP = CP.at[0, 0].set(x0)
    CP = CP.at[0, 1].set(x0 + (T0 / 5.0) * v0)
    CP = CP.at[0, 2].set(x0 + (2 * T0 / 5.0) * v0 + (T0**2 / 20.0) * a0)

    # Inject end boundary conditions
    TM = T[-1]
    CP = CP.at[-1, 5].set(xf)
    CP = CP.at[-1, 4].set(xf - (TM / 5.0) * vf)
    CP = CP.at[-1, 3].set(xf - (2 * TM / 5.0) * vf + (TM**2 / 20.0) * af)

    # Enforce C2 continuity at segment junctions
    for s in range(M - 1):
        Ts, Ts1 = T[s], T[s + 1]
        r = Ts / (Ts1 + eps)
        P50 = CP[s + 1, 0]
        CP = CP.at[s, 5].set(P50)
        delta01 = CP[s + 1, 1] - CP[s + 1, 0]
        CP = CP.at[s, 4].set(P50 - r * delta01)
        sec = CP[s + 1, 2] - 2 * CP[s + 1, 1] + CP[s + 1, 0]
        P4 = CP[s, 4]
        CP = CP.at[s, 3].set(2 * P4 - P50 + r**2 * sec)

    return CP, T


def sample_trajectory_jax(CP, T, N_per_seg):
    """
    Sample positions, velocities, and accelerations WITHOUT searchsorted.
    Uses per-segment loops for differentiability.

    Args:
        CP: (M, 6, 3) control points per segment
        T: (M,) segment durations
        N_per_seg: int, samples per segment

    Returns:
        positions: (M * N_per_seg, 3)
        velocities: (M * N_per_seg, 3)
        accelerations: (M * N_per_seg, 3)
        times: (M * N_per_seg,) absolute times for each sample
    """
    M = CP.shape[0]
    all_pos = []
    all_vel = []
    all_acc = []
    all_times = []

    t_cumsum = 0.0
    for s in range(M):  # Python loop - unrolled at trace time
        Ts = T[s]
        # Sample at midpoints within segment
        us = jnp.linspace(0.0, 1.0, N_per_seg, endpoint=False) + 0.5 / N_per_seg

        for j in range(N_per_seg):  # Python loop
            u = us[j]
            um = 1.0 - u

            # Position using degree-5 Bernstein basis
            pos = jnp.zeros(3)
            for i in range(6):
                B_i = bernstein5_val(u, i)
                pos = pos + B_i * CP[s, i, :]
            all_pos.append(pos)

            # Velocity: (5/T[s]) * derivative
            # Derivative of degree-5 Bernstein → degree-4 differences
            vel = jnp.zeros(3)
            for i in range(5):
                B4_i = bernstein4_val(u, i)
                vel = vel + B4_i * (CP[s, i + 1, :] - CP[s, i, :])
            vel = (5.0 / Ts) * vel
            all_vel.append(vel)

            # Acceleration: (20/T[s]²) * second derivative
            # Second derivative → degree-3 second differences
            acc = jnp.zeros(3)
            for i in range(4):
                B3_i = bernstein3_val(u, i)
                diff2 = CP[s, i + 2, :] - 2 * CP[s, i + 1, :] + CP[s, i, :]
                acc = acc + B3_i * diff2
            acc = (20.0 / (Ts**2)) * acc
            all_acc.append(acc)

            # Absolute time for this sample
            t_abs = t_cumsum + u * Ts
            all_times.append(t_abs)

        t_cumsum += Ts

    return jnp.stack(all_pos), jnp.stack(all_vel), jnp.stack(all_acc), jnp.stack(all_times)


# Legacy function kept for compatibility but not used in objective
def eval_traj_jax(CP, T, N_samp):
    """
    Legacy trajectory sampling function (uses searchsorted - not differentiable).
    Use sample_trajectory_jax instead for gradient computation.
    """
    M = CP.shape[0]
    N_per_seg = int(jnp.maximum(1, N_samp // M))
    pts, _, _, _ = sample_trajectory_jax(CP, T, N_per_seg)
    return pts


def fit_quintic_jax(x0, v0, a0, xf, vf, af, T):
    """
    Fit quintic polynomial trajectory for dynamic obstacle.

    Args:
        x0, v0, a0: Start pos/vel/acc (3,)
        xf, vf, af: End pos/vel/acc (3,)
        T: Duration (scalar)

    Returns:
        cx, cy, cz: Polynomial coefficients (6,) each
    """
    # Build constraint matrix for quintic polynomial
    A = jnp.zeros((6, 6))

    # p(0) = x0
    A = A.at[0, 0].set(1.0)

    # p'(0) = v0
    A = A.at[1, 1].set(1.0)

    # p''(0) = a0
    A = A.at[2, 2].set(2.0)

    # p(T) = xf
    for i in range(6):
        A = A.at[3, i].set(T**i)

    # p'(T) = vf
    for i in range(1, 6):
        A = A.at[4, i].set(i * T**(i - 1))

    # p''(T) = af
    for i in range(2, 6):
        A = A.at[5, i].set(i * (i - 1) * T**(i - 2))

    # Solve for each dimension
    bx = jnp.array([x0[0], v0[0], a0[0], xf[0], vf[0], af[0]])
    by = jnp.array([x0[1], v0[1], a0[1], xf[1], vf[1], af[1]])
    bz = jnp.array([x0[2], v0[2], a0[2], xf[2], vf[2], af[2]])

    cx = jnp.linalg.solve(A, bx)
    cy = jnp.linalg.solve(A, by)
    cz = jnp.linalg.solve(A, bz)

    return cx, cy, cz


def f_obs_poly_jax(t, cx, cy, cz):
    """
    Evaluate obstacle polynomial trajectory at given times.

    Args:
        t: Time values (N,) or scalar
        cx, cy, cz: Polynomial coefficients (6,) each

    Returns:
        positions: Obstacle positions (N, 3) or (3,) if t is scalar
    """
    px = sum(cx[i] * t**i for i in range(6))
    py = sum(cy[i] * t**i for i in range(6))
    pz = sum(cz[i] * t**i for i in range(6))
    return jnp.stack([px, py, pz], axis=-1)


# =============================================================================
# Cost Term Functions
# =============================================================================

def compute_J_time_jax(T):
    """
    Time cost: sum of segment durations.

    Args:
        T: Segment durations (M,)

    Returns:
        J_time: Total trajectory time
    """
    return jnp.sum(T)


def compute_J_jerk_jax(CP, T):
    """
    Jerk cost: sum of third-difference norms weighted by segment time.

    Args:
        CP: Control points (M, 6, 3)
        T: Segment durations (M,)

    Returns:
        J_jerk: Jerk cost
    """
    M = CP.shape[0]
    J = 0.0
    eps = 1e-10  # Numerical stability
    for s in range(M):
        Ts = T[s]
        # Third differences: Δ³P[s,m] = P[s,m+3] - 3*P[s,m+2] + 3*P[s,m+1] - P[s,m]
        for m in range(3):
            D3 = CP[s, m + 3] - 3 * CP[s, m + 2] + 3 * CP[s, m + 1] - CP[s, m]
            J += (3600.0 / (Ts**5 + eps)) * jnp.dot(D3, D3)
    return J


def compute_J_acc_jax(CP, T):
    """
    Acceleration cost: sum of second-difference norms weighted by segment time.

    Args:
        CP: Control points (M, 6, 3)
        T: Segment durations (M,)

    Returns:
        J_acc: Acceleration cost
    """
    M = CP.shape[0]
    J = 0.0
    eps = 1e-10  # Numerical stability
    for s in range(M):
        Ts = T[s]
        # Second differences: Δ²P[s,m] = P[s,m+2] - 2*P[s,m+1] + P[s,m]
        for m in range(4):
            D2 = CP[s, m + 2] - 2 * CP[s, m + 1] + CP[s, m]
            J += (400.0 / (Ts**3 + eps)) * jnp.dot(D2, D2)
    return J


def compute_J_dyn_jax(CP, T, env_params):
    """
    Dynamic obstacle avoidance cost using ellipsoid distance metric.
    Uses per-segment sampling for differentiability.

    Args:
        CP: Control points (M, 6, 3)
        T: Segment durations (M,)
        env_params: Environment parameters dict

    Returns:
        J_dyn: Dynamic obstacle cost
    """
    obstacles = env_params['obstacles']
    N_per_seg = env_params.get('N_per_seg', 3)
    Cw = env_params['Cw']
    c = env_params['c_ellipsoid']

    if len(obstacles) == 0:
        return 0.0

    # Sample ego trajectory using differentiable method
    pts, _, _, t_abs = sample_trajectory_jax(CP, T, N_per_seg)

    J = 0.0

    # Ellipsoid transformation matrix
    E_half = jnp.diag(jnp.array([1.0, 1.0, 1.0 / jnp.sqrt(c)]))

    for obs in obstacles:
        # Fit obstacle trajectory
        cx, cy, cz = fit_quintic_jax(
            obs['x0'], obs['v0'], obs['a0'],
            obs['xf'], obs['vf'], obs['af'],
            obs['T']
        )

        # Evaluate obstacle positions at matching times
        k_pts = f_obs_poly_jax(t_abs, cx, cy, cz)

        # Ellipsoid distance
        diff = (pts - k_pts) @ E_half.T
        d2 = jnp.sum(diff**2, axis=1)

        # Penalty: max(Cw² - d², 0)³
        violation = Cw**2 - d2
        J += jnp.sum(jnp.clip(violation, 0.0, None)**3)

    return J


def compute_J_stat_jax(CP, env_params):
    """
    Static obstacle avoidance cost using half-space constraints.

    Args:
        CP: Control points (M, 6, 3)
        env_params: Environment parameters dict with keys:
            - 'A_stat': List of constraint matrices per segment
            - 'b_stat': List of constraint offsets per segment
            - 'Co': Static obstacle clearance

    Returns:
        J_stat: Static obstacle cost
    """
    A_stat = env_params['A_stat']
    b_stat = env_params['b_stat']
    Co = env_params['Co']

    M = CP.shape[0]
    J = 0.0

    for seg in range(M):
        for idx in range(6):
            P = CP[seg, idx]

            # Determine which segment constraints apply
            # CP[s,0] is shared between segment s-1 and s
            if idx == 0 and seg > 0:
                planes = [seg - 1, seg]
            else:
                planes = [seg]

            for p in planes:
                A = A_stat[p]
                b = b_stat[p]

                # Skip if no constraints
                if A.shape[0] == 0:
                    continue

                # Signed distance: h = A·P - b
                h = A @ P - b  # (n_planes,)

                # Cubic penalty: max(Co - h, 0)³
                violation = Co - h
                J += jnp.sum(jnp.clip(violation, 0.0, None)**3)

    return J


def compute_J_vel_constr_jax(CP, T, env_params):
    """
    Velocity constraint violation cost.

    Args:
        CP: Control points (M, 6, 3)
        T: Segment durations (M,)
        env_params: Environment parameters dict with keys:
            - 'V_max': Maximum velocity
            - 'N_samp': Number of trajectory samples

    Returns:
        J_vel: Velocity violation cost
    """
    V_max = env_params['V_max']
    N_samp = env_params['N_samp']

    # Sample velocities
    _, vels, _ = eval_traj_and_derivs_jax(CP, T, N_samp)
    vel_norms = jnp.linalg.norm(vels, axis=1)

    # Penalty: max(||v|| - V_max, 0)³
    violation = vel_norms - V_max
    J = jnp.sum(jnp.clip(violation, 0.0, None)**3)

    return J


def compute_J_acc_constr_jax(CP, T, env_params):
    """
    Acceleration constraint violation cost.

    Args:
        CP: Control points (M, 6, 3)
        T: Segment durations (M,)
        env_params: Environment parameters dict with keys:
            - 'A_max': Maximum acceleration
            - 'N_samp': Number of trajectory samples

    Returns:
        J_acc: Acceleration violation cost
    """
    A_max = env_params['A_max']
    N_samp = env_params['N_samp']

    # Sample accelerations
    _, _, accs = eval_traj_and_derivs_jax(CP, T, N_samp)
    acc_norms = jnp.linalg.norm(accs, axis=1)

    # Penalty: max(||a|| - A_max, 0)³
    violation = acc_norms - A_max
    J = jnp.sum(jnp.clip(violation, 0.0, None)**3)

    return J


# =============================================================================
# Main Objective Function
# =============================================================================

def evaluate_objective_jax(z, theta, env_params):
    """
    Fully differentiable MIGHTY cost function.

    Args:
        z: (K,) decision vector [free_CPs (F×3), σ_slacks (M)]
        theta: dict of neural network outputs that parameterize the cost:
            'time_weight': scalar
            'dyn_weight': scalar
            'stat_weight': scalar
            'accel_weight': scalar
            'jerk_weight': scalar
            'vel_constr_weight': scalar
            'acc_constr_weight': scalar
        env_params: dict of fixed environment parameters:
            'x0', 'v0', 'a0': (3,) start boundary conditions
            'xf', 'vf', 'af': (3,) end boundary conditions
            'wps': (M+1, 3) global waypoints
            'free_cps_indices': list of (seg, cp_idx) tuples
            'obstacles': list of obstacle dicts
            'A_stat', 'b_stat': static constraint planes
            'V_max', 'A_max': dynamic limits
            'N_per_seg': int, samples per segment (for JIT compatibility)
            'Co', 'Cw', 'c_ellipsoid': clearance params

    Returns:
        scalar cost J(z, theta)
    """
    # Unpack environment parameters
    x0 = env_params['x0']
    v0 = env_params['v0']
    a0 = env_params['a0']
    xf = env_params['xf']
    vf = env_params['vf']
    af = env_params['af']
    wps = env_params['wps']
    free_cps_indices = env_params['free_cps_indices']

    # Get N_per_seg (must be Python int, not traced value, for range() loops)
    N_per_seg = env_params.get('N_per_seg', 3)  # Default: 3 samples per segment

    # Reconstruct control points and segment times
    CP, T = reconstruct_jax(z, x0, v0, a0, xf, vf, af, wps, free_cps_indices)

    # Compute cost terms that don't need trajectory sampling
    J_time = compute_J_time_jax(T)
    J_stat = compute_J_stat_jax(CP, env_params)
    J_acc = compute_J_acc_jax(CP, T)
    J_jerk = compute_J_jerk_jax(CP, T)

    # Sample trajectory once using differentiable method
    pts, vels, accs, t_abs = sample_trajectory_jax(CP, T, N_per_seg)

    # Dynamic obstacle cost (uses sampled positions and times)
    J_dyn = compute_J_dyn_jax(CP, T, env_params)

    # Velocity constraint
    vel_norms = jnp.linalg.norm(vels, axis=1)
    violation_vel = vel_norms - env_params['V_max']
    J_vel_constr = jnp.sum(jnp.clip(violation_vel, 0.0, None)**3)

    # Acceleration constraint
    acc_norms = jnp.linalg.norm(accs, axis=1)
    violation_acc = acc_norms - env_params['A_max']
    J_acc_constr = jnp.sum(jnp.clip(violation_acc, 0.0, None)**3)

    # Weighted sum using neural network-provided weights
    return (theta['time_weight'] * J_time
            + theta['dyn_weight'] * J_dyn
            + theta['stat_weight'] * J_stat
            + theta['accel_weight'] * J_acc
            + theta['jerk_weight'] * J_jerk
            + theta['vel_constr_weight'] * J_vel_constr
            + theta['acc_constr_weight'] * J_acc_constr)
