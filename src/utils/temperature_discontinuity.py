import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from scipy.ndimage import maximum_filter1d, minimum_filter1d

from src.core.geometry import DomainGeometry
from src.parameters.config import ExperimentConfig
from src.parameters.material_properties import MaterialProperties
from tests.numerical_experiments.one_dim.analytic_solution_1d_2ph import (
    get_ice_temp,
    get_water_temp,
    calculate_gamma,
)
from src.core.constants import ABS_ZERO


geometry = DomainGeometry(
    width=1.0,
    height=1.0,
    end_time=60.0 * 60.0 * 24.0,
    n_x=5,
    n_y=301,
    n_t=60 * 24,
)

material_props = MaterialProperties(
    u_pt=273.15,
    specific_heat_liquid=4120.7,
    specific_heat_solid=2056.8,
    specific_latent_heat=3.33e5,
    density_liquid=999.84,
    density_solid=918.9,
    thermal_conductivity_liquid=0.59,
    thermal_conductivity_solid=2.21,
    kinematic_viscosity_coeffs=[
        0.000108963453,
        -9.28722151e-07,
        2.65889022e-09,
        -2.54761652e-12,
    ],
    volumetric_thermal_exp_coeffs=[7.68e-6],
)

max_temp = 278.15
min_temp = 268.15

cfg = ExperimentConfig(
    geometry=geometry,
    material_props=material_props,
    u_ref=0.5 * (min_temp + max_temp),
    delta_u=0.5 * (max_temp - min_temp),
    v=0.01,
    l=geometry.max_dimension,
    u_solid=273.0,
    u_liquid=273.3,
    epsilon=1e-6,
    save_interval=1,
)

gamma = calculate_gamma(cfg=cfg, min_temp=min_temp, max_temp=max_temp)
t = geometry.end_time
s = t**0.5 * gamma  # interface position

print(f"True interface position: s = {s:.6f}")
u_star = material_props.u_pt + ABS_ZERO


def locate_interface_uniform(xs, u, u_pt):
    thetas = u - u_pt
    i = np.where(thetas[:-1] * thetas[1:] < 0)[0][0]
    h = xs[1] - xs[0]

    # Linear interpolation (your current method is correct)
    s_lin = xs[i] + h * (u_pt - u[i]) / (u[i + 1] - u[i])

    # True piecewise: use left-sided slope for left piece
    if i > 0:
        m_left = (u[i] - u[i - 1]) / h
        # Solve: u[i] + m_left * (s - xs[i]) = u_pt
        s_left_piece = xs[i] + (u_pt - u[i]) / m_left
    else:
        s_left_piece = s_lin

    # Use right-sided slope for right piece
    if i + 2 < len(u):
        m_right = (u[i + 2] - u[i + 1]) / h
        # Solve: u[i+1] + m_right * (s - xs[i+1]) = u_pt
        s_right_piece = xs[i + 1] + (u_pt - u[i + 1]) / m_right
    else:
        s_right_piece = s_lin

    # Choose the piece that gives interface position within the interval
    if xs[i] <= s_left_piece <= xs[i + 1]:
        s_piecewise = s_left_piece
    elif xs[i] <= s_right_piece <= xs[i + 1]:
        s_piecewise = s_right_piece
    else:
        s_piecewise = s_lin  # fallback

    return s_lin, s_piecewise


def calculate_envelope(h_values, errors, window_size=20):
    """Calculate upper and lower envelopes using local maxima/minima"""
    # Convert to numpy arrays
    h_arr = np.array(h_values)
    err_arr = np.array(errors)

    # Use maximum/minimum filters to find local extrema
    upper_envelope = maximum_filter1d(err_arr, size=window_size)
    lower_envelope = minimum_filter1d(err_arr, size=window_size)

    return h_arr, upper_envelope, lower_envelope


def fit_power_law(h_values, errors, start_idx=None, end_idx=None):
    """Fit power law: error = C * h^alpha using log-log regression"""
    h_arr = np.array(h_values[start_idx:end_idx])
    err_arr = np.array(errors[start_idx:end_idx])

    # Remove zeros and negative values
    valid_mask = (err_arr > 0) & (h_arr > 0)
    h_clean = h_arr[valid_mask]
    err_clean = err_arr[valid_mask]

    # Log-log linear regression
    log_h = np.log(h_clean)
    log_err = np.log(err_clean)

    # Fit: log(err) = log(C) + alpha * log(h)
    coeffs = np.polyfit(log_h, log_err, 1)
    alpha = coeffs[0]
    log_C = coeffs[1]
    C = np.exp(log_C)

    # Calculate R-squared
    log_err_pred = log_C + alpha * log_h
    ss_res = np.sum((log_err - log_err_pred) ** 2)
    ss_tot = np.sum((log_err - np.mean(log_err)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)

    return alpha, C, r_squared


def check_derivative_jump():
    """Verify that analytical solution has derivative discontinuity"""
    eps = 1e-8

    # Temperatures just before and after interface
    T_before = get_ice_temp(s - eps, gamma, t, min_temp + ABS_ZERO, material_props)
    T_at_interface_ice = get_ice_temp(s, gamma, t, min_temp + ABS_ZERO, material_props)
    T_at_interface_water = get_water_temp(
        s, gamma, t, max_temp + ABS_ZERO, material_props
    )
    T_after = get_water_temp(s + eps, gamma, t, max_temp + ABS_ZERO, material_props)

    # Calculate gradients
    grad_ice = (T_at_interface_ice - T_before) / eps
    grad_water = (T_after - T_at_interface_water) / eps

    print(f"Temperature at interface (ice side): {T_at_interface_ice:.6f}")
    print(f"Temperature at interface (water side): {T_at_interface_water:.6f}")
    print(
        f"Temperature continuity: {abs(T_at_interface_ice - T_at_interface_water):.2e}"
    )
    print(f"Gradient in ice: {grad_ice:.3f}")
    print(f"Gradient in water: {grad_water:.3f}")
    print(f"Gradient jump: {grad_water - grad_ice:.3f}")
    print()


# Check derivative jump
check_derivative_jump()

# ---- GRID REFINEMENT TEST ----
n_values = list(range(50, 800, 5))  # Denser sampling for better envelope
h_values = []
errors_lin = []
errors_pw = []
xi_values = []  # Track relative position in cell

print("Running grid refinement test...")
for n_y in n_values:
    xs = np.linspace(0, geometry.height, n_y)
    h = xs[1] - xs[0]
    h_values.append(h)

    # Calculate which cell contains the interface
    cell_index = int(s / h)
    xi = (s - xs[cell_index]) / h if cell_index < len(xs) - 1 else 0.5
    xi_values.append(xi)

    # Analytical temperatures
    Ts = []
    for y in xs:
        if y <= s:
            Ts.append(
                get_ice_temp(
                    y=y,
                    gamma=gamma,
                    t=t,
                    min_temp=min_temp + ABS_ZERO,
                    material_props=material_props,
                )
            )
        else:
            Ts.append(
                get_water_temp(
                    y=y,
                    gamma=gamma,
                    t=t,
                    max_temp=max_temp + ABS_ZERO,
                    material_props=material_props,
                )
            )
    Ts = np.array(Ts)

    # Reconstruct interface
    try:
        s_lin, s_pw = locate_interface_uniform(xs, Ts, u_star)
        errors_lin.append(abs(s_lin - s))
        errors_pw.append(abs(s_pw - s))
    except:
        # Remove this point if reconstruction fails
        h_values.pop()
        xi_values.pop()

print(f"Generated {len(h_values)} data points")

# Calculate envelopes
h_env, upper_env_lin, lower_env_lin = calculate_envelope(h_values, errors_lin)
_, upper_env_pw, lower_env_pw = calculate_envelope(h_values, errors_pw)

# Fit power laws to envelopes and raw data
alpha_lin_raw, C_lin_raw, r2_lin_raw = fit_power_law(h_values, errors_lin)
alpha_pw_raw, C_pw_raw, r2_pw_raw = fit_power_law(h_values, errors_pw)

alpha_lin_env, C_lin_env, r2_lin_env = fit_power_law(h_values, upper_env_lin)
alpha_pw_env, C_pw_env, r2_pw_env = fit_power_law(h_values, upper_env_pw)

print("\n" + "=" * 60)
print("CONVERGENCE ANALYSIS RESULTS")
print("=" * 60)
print(
    f"Linear interpolation (raw data):     α = {alpha_lin_raw:.3f}, R² = {r2_lin_raw:.3f}"
)
print(
    f"Piecewise method (raw data):         α = {alpha_pw_raw:.3f}, R² = {r2_pw_raw:.3f}"
)
print(
    f"Linear interpolation (envelope):     α = {alpha_lin_env:.3f}, R² = {r2_lin_env:.3f}"
)
print(
    f"Piecewise method (envelope):         α = {alpha_pw_env:.3f}, R² = {r2_pw_env:.3f}"
)
print()
print(
    "Expected: Linear method α ≈ 1.0, Piecewise method α ≈ 2.0 (if hypothesis is correct)"
)
print("=" * 60)

# ---- PLOTTING ----
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Raw convergence data (log-log)
ax1.loglog(
    h_values, errors_lin, "o-", alpha=0.7, markersize=2, label="Linear interpolation"
)
ax1.loglog(
    h_values, errors_pw, "s-", alpha=0.7, markersize=2, label="Piecewise one-sided"
)

# Add fitted lines
h_fit = np.logspace(np.log10(min(h_values)), np.log10(max(h_values)), 100)
ax1.loglog(
    h_fit,
    C_lin_raw * h_fit**alpha_lin_raw,
    "--",
    color="blue",
    alpha=0.8,
    label=f"Linear fit: h^{alpha_lin_raw:.2f}",
)
ax1.loglog(
    h_fit,
    C_pw_raw * h_fit**alpha_pw_raw,
    "--",
    color="red",
    alpha=0.8,
    label=f"Piecewise fit: h^{alpha_pw_raw:.2f}",
)

ax1.set_xlabel("Grid spacing h")
ax1.set_ylabel("Error in interface position")
ax1.set_title("Raw Convergence Data (Log-Log)")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Envelope analysis (log-log)
ax2.loglog(
    h_values,
    upper_env_lin,
    "-",
    color="blue",
    linewidth=2,
    label="Linear upper envelope",
)
ax2.loglog(
    h_values,
    upper_env_pw,
    "-",
    color="red",
    linewidth=2,
    label="Piecewise upper envelope",
)

# Add envelope fits
ax2.loglog(
    h_fit,
    C_lin_env * h_fit**alpha_lin_env,
    "--",
    color="navy",
    alpha=0.8,
    label=f"Linear env fit: h^{alpha_lin_env:.2f}",
)
ax2.loglog(
    h_fit,
    C_pw_env * h_fit**alpha_pw_env,
    "--",
    color="darkred",
    alpha=0.8,
    label=f"Piecewise env fit: h^{alpha_pw_env:.2f}",
)

ax2.set_xlabel("Grid spacing h")
ax2.set_ylabel("Error in interface position")
ax2.set_title("Envelope Analysis (Log-Log)")
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Linear scale to see oscillations
ax3.plot(
    h_values, errors_lin, "o-", alpha=0.7, markersize=2, label="Linear interpolation"
)
ax3.plot(
    h_values, errors_pw, "s-", alpha=0.7, markersize=2, label="Piecewise one-sided"
)
ax3.plot(
    h_values,
    upper_env_lin,
    "-",
    color="blue",
    linewidth=2,
    alpha=0.8,
    label="Linear envelope",
)
ax3.plot(
    h_values,
    upper_env_pw,
    "-",
    color="red",
    linewidth=2,
    alpha=0.8,
    label="Piecewise envelope",
)

ax3.set_xlabel("Grid spacing h")
ax3.set_ylabel("Error in interface position")
ax3.set_title("Oscillations and Envelopes (Linear Scale)")
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Error vs relative position in cell
ax4.scatter(xi_values, errors_lin, alpha=0.6, s=10, label="Linear interpolation")
ax4.scatter(xi_values, errors_pw, alpha=0.6, s=10, label="Piecewise one-sided")
ax4.set_xlabel("Relative position in cell (ξ)")
ax4.set_ylabel("Error in interface position")
ax4.set_title("Error vs Interface Position in Cell")
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Additional analysis: Check if envelope is truly linear/parabolic
print("\n" + "=" * 60)
print("ENVELOPE SHAPE ANALYSIS")
print("=" * 60)

# Sample points for envelope analysis
sample_indices = np.arange(0, len(h_values), 5)
h_sample = np.array(h_values)[sample_indices]
env_lin_sample = upper_env_lin[sample_indices]
env_pw_sample = upper_env_pw[sample_indices]

# Fit different models to envelopes
print("Linear interpolation envelope:")
print(f"  Power law fit: error ∝ h^{alpha_lin_env:.3f}")
if abs(alpha_lin_env - 1.0) < 0.1:
    print("  ✓ Consistent with linear envelope (O(h))")
else:
    print(f"  ⚠ Deviates from linear envelope by {abs(alpha_lin_env - 1.0):.3f}")

print("\nPiecewise method envelope:")
print(f"  Power law fit: error ∝ h^{alpha_pw_env:.3f}")
if abs(alpha_pw_env - 2.0) < 0.1:
    print("  ✓ Consistent with parabolic envelope (O(h²))")
elif abs(alpha_pw_env - 1.0) < 0.1:
    print("  ~ Similar to linear envelope (O(h))")
else:
    print(f"  ? Envelope shows h^{alpha_pw_env:.3f} behavior")

print("=" * 60)
