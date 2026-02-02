"""Debug free CP indices mapping."""

import numpy as np
from mighty_solver_test import create_test_scenario, create_initial_guess

env_params, theta_default, M, F = create_test_scenario()
z0 = create_initial_guess(M, F, np.array(env_params['wps']))

print("Free CP Indices Mapping")
print("=" * 70)

print(f"Number of segments (M): {M}")
print(f"Number of free CPs (F): {F}")
print()

print("Free CP indices (seg, cp_idx):")
for i, (seg, cp_idx) in enumerate(env_params['free_cps_indices']):
    print(f"  Free CP {i}: segment {seg}, index {cp_idx}")
print()

print("Gradient NaN pattern:")
print("  NaN: Free CPs 4, 5 -> segments 1, 1")
print("  NaN: Time slacks 1, 2")
print()

print("This suggests:")
print("  - Free CPs in segment 1 (indices 4, 5) have gradient issues")
print("  - Time slacks for segments 1, 2 also have issues")
print("  - Segments > 0 have gradient flow problems")
print()

print("Checking waypoints:")
wps = env_params['wps']
for i, wp in enumerate(wps):
    print(f"  WP {i}: {wp}")
print()

print("The issue might be related to the per-segment sampling loop.")
print("Let me check if segment 0 works differently from segments 1, 2...")
