import numpy as np
from scipy import optimize
from scipy.special import erf, erfc
import matplotlib.pyplot as plt
from typing import Tuple
import warnings
from dataclasses import dataclass


@dataclass
class StefanParameters:
    """Parameters for the 2D Stefan problem"""

    beta: float  # Latent to sensible heat ratio beta = L/[c_s(T_f - T_w)]
    Ti_star: float  # Dimensionless initial temperature T^*_i = (T_i - T_f)/(T_f - T_w)
    alpha_ratio: float = (
        1.0  # Thermal diffusivity ratio alpha_s/alpha_l (assumed = 1 in paper)
    )


class StefanCornerSolver:
    """
    Analytical solver for 2D Stefan problem in a corner
    Based on Rathjen & Jiji (1971) ASME Journal of Heat Transfer
    """

    def __init__(self, params: StefanParameters):
        # Validate parameters
        if params.beta <= 0:
            raise ValueError(f"beta must be > 0, got beta={params.beta}")
        if params.Ti_star <= -1:
            raise ValueError(f"Ti_star must be > -1, got Ti_star={params.Ti_star}")
        if params.alpha_ratio <= 0:
            raise ValueError(
                f"alpha_ratio must be > 0, got alpha_ratio={params.alpha_ratio}"
            )

        self.params = params
        self.beta = params.beta
        self.Ti_star = params.Ti_star

        # Gauss quadrature points and weights (20-point)
        self.gauss_points, self.gauss_weights = self._get_gauss_legendre_20()

        # Calculate lambda (1D Neumann solution parameter)
        self.lambda_val = self._solve_neumann_lambda()

        # Interface parameters (to be solved)
        self.x0_star = None
        self.x1_star = None
        self.C = None
        self.m = None

        # Set integration upper limit A
        if self.Ti_star > 1.0 and self.beta > 1.0:
            self.A = 5 * self.lambda_val
        else:
            self.A = 3 * self.lambda_val

    @staticmethod
    def _get_gauss_legendre_20():
        """20-point Gauss-Legendre quadrature points and weights for [-1,1]."""
        points_half = np.array(
            [
                0.0765265,
                0.2277859,
                0.3737061,
                0.5108670,
                0.6360537,
                0.7463319,
                0.8391170,
                0.9122344,
                0.9639719,
                0.9931286,
            ]
        )

        weights_half = np.array(
            [
                0.1527534,
                0.1491730,
                0.1420961,
                0.1316886,
                0.1181945,
                0.1019301,
                0.0832767,
                0.0626720,
                0.0406014,
                0.0176140,
            ]
        )

        points = np.concatenate((-points_half[::-1], points_half))
        weights = np.concatenate((weights_half[::-1], weights_half))

        return points, weights

    @staticmethod
    def _transform_interval(xi, a, b):
        """Transform Gauss points from [-1,1] to [a,b]."""
        return 0.5 * ((b - a) * xi + (a + b))

    @staticmethod
    def _transform_weight(w, a, b):
        """Transform Gauss weights from [-1,1] to [a,b]."""
        return 0.5 * (b - a) * w

    def _solve_neumann_lambda(self) -> float:
        """
        Solve equation (15) for lambda (Neumann's 1D solution parameter)
        exp(-lambda^2)/erf(lambda) - (Ti_star exp(-lambda^2))/erfc(lambda) = sqrt(pi) beta lambda
        """

        def neumann_eq(lam):
            if lam <= 0:
                return float("inf")
            try:
                exp_neg_lam2 = np.exp(-(lam**2))
                erf_lam = erf(lam)
                erfc_lam = erfc(lam)

                if abs(erf_lam) < 1e-15 or abs(erfc_lam) < 1e-15:
                    return float("inf")

                lhs = exp_neg_lam2 / erf_lam - (self.Ti_star * exp_neg_lam2) / erfc_lam
                rhs = self.beta * np.sqrt(np.pi) * lam
                return lhs - rhs
            except (ZeroDivisionError, OverflowError):
                return float("inf")

        try:
            result = optimize.root_scalar(
                neumann_eq,
                bracket=[0.01, 5.0],
                method="brentq",
                xtol=1e-12,
            )
            return result.root
        except ValueError:
            print(
                f"Bracket method failed for beta={self.beta}, Ti_star={self.Ti_star}. Using optimization."
            )
            result = optimize.minimize_scalar(
                lambda x: abs(neumann_eq(x)), bounds=(0.01, 5.0), method="bounded"
            )
            return result.x

    def _superhyperbola(self, x_star: float, c: float, m: float) -> float:
        """
        Superhyperbola interface shape f(x_star) = [ lambda^m + C / (x_star^m - lambda^m) ]^(1/m)
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            num = self.lambda_val**m + c / (x_star**m - self.lambda_val**m)
            return num ** (1.0 / m)

    def _superhyperbola_derivative(self, x_star: float, c: float, m: float) -> float:
        """
        Derivative of superhyperbola interface shape
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            A = x_star**m - self.lambda_val**m
            inner = self.lambda_val**m + c / A
            # d/dx of inner^(1/m) = (1/m)*inner^(1/m - 1) * d(inner)/dx
            d_inner_dx = -c * m * x_star ** (m - 1) / (A**2)
            return (1.0 / m) * inner ** (1 / m - 1) * d_inner_dx

    @staticmethod
    def _k_function(eta: float, tau: float, x_star: float) -> float:
        """K function from equation (14)"""
        exp1 = np.exp(-((x_star - eta * np.sqrt(tau)) ** 2) / (1 - tau))
        exp2 = np.exp(-((x_star + eta * np.sqrt(tau)) ** 2) / (1 - tau))
        return exp1 - exp2

    @staticmethod
    def _e_function(tau: float, a: float, x_star: float) -> float:
        """E function from equation (16)"""
        sqrt_term = np.sqrt(1 - tau)
        arg1 = (x_star - a * np.sqrt(tau)) / sqrt_term
        arg2 = (x_star + a * np.sqrt(tau)) / sqrt_term
        return erf(arg1) + erf(arg2)

    def _compute_v_integral(self, x_star, y_star, x0_star, x1_star, c, m):
        """
        Compute the V integral from equation (16).
        """
        # Setup integration intervals
        # τ intervals: [0, 0.9] and [0.9, 1.0]
        tau_intervals = [(0.0, 0.9), (0.9, 1.0)]
        # η intervals: [x0_star, x1_star] and [x1_star, A]
        eta_intervals = [(x0_star, x1_star), (x1_star, self.A)]

        total_integral = 0.0

        # Double integration over all interval combinations
        for tau_a, tau_b in tau_intervals:
            for eta_a, eta_b in eta_intervals:
                # Transform Gauss points and weights to current intervals
                tau_points = self._transform_interval(self.gauss_points, tau_a, tau_b)
                tau_weights = self._transform_weight(self.gauss_weights, tau_a, tau_b)

                eta_points = self._transform_interval(self.gauss_points, eta_a, eta_b)
                eta_weights = self._transform_weight(self.gauss_weights, eta_a, eta_b)

                # Double sum over quadrature points
                for i, (tau_i, w_tau_i) in enumerate(zip(tau_points, tau_weights)):
                    for j, (eta_j, w_eta_j) in enumerate(zip(eta_points, eta_weights)):
                        # Avoid singularity at tau=1, eta=x_star
                        if abs(tau_i - 1.0) < 1e-10 and abs(eta_j - x_star) < 1e-10:
                            continue

                        f_eta = self._superhyperbola(eta_j, c, m)
                        df_deta = self._superhyperbola_derivative(eta_j, c, m)

                        numerator = f_eta - eta_j * df_deta
                        denominator = 1.0 - tau_i

                        if abs(denominator) < 1e-15:
                            continue

                        # Two K function terms
                        k1 = self._k_function(eta_j, tau_i, x_star)
                        k2 = self._k_function(f_eta, tau_i, y_star)
                        k3 = self._k_function(f_eta, tau_i, x_star)
                        k4 = self._k_function(eta_j, tau_i, y_star)

                        integrand = (numerator / denominator) * (k1 * k2 + k3 * k4)

                        total_integral += integrand * w_tau_i * w_eta_j

        # Add the Lambda contribution (second integral in equation 16)
        lambda_integral = 0.0
        for tau_a, tau_b in tau_intervals:
            tau_points = self._transform_interval(self.gauss_points, tau_a, tau_b)
            tau_weights = self._transform_weight(self.gauss_weights, tau_a, tau_b)

            for tau_i, w_tau_i in zip(tau_points, tau_weights):
                denominator = np.sqrt(tau_i) * np.sqrt(1 - tau_i)

                k_lambda_y = self._k_function(self.lambda_val, tau_i, y_star)
                k_lambda_x = self._k_function(self.lambda_val, tau_i, x_star)
                e_lambda_x = self._e_function(tau_i, self.A, x_star)
                e_lambda_y = self._e_function(tau_i, self.A, y_star)

                integrand = (
                    k_lambda_y * e_lambda_x + k_lambda_x * e_lambda_y
                ) / denominator
                lambda_integral += integrand * w_tau_i

        return self.beta * (
            total_integral / (2 * np.pi)
            + self.lambda_val * lambda_integral / (4 * np.sqrt(np.pi))
        )

    def _interface_equation(
        self, x_star: float, x0_star: float, x1_star: float, c: float, m: float
    ) -> float:
        """
        Equation (17) - the main interface equation to be satisfied
        """
        # U part (equation 12)
        f_x = self._superhyperbola(x_star, c, m)
        u_part = -1 + (1 + self.Ti_star) * erf(x_star) * erf(f_x)

        # V part (equation 16)
        v_part = self._compute_v_integral(x_star, f_x, x0_star, x1_star, c, m)

        return u_part + v_part

    def solve_interface_parameters(self) -> Tuple[float, float, float, float]:
        """
        Solve for x0*, x1*, C, and m parameters
        This requires solving the nonlinear system from equation (17)
        """

        def system_equations(params):
            x0_star, x1_star = params
            # Calculate C and m from constraints
            # f(x0*) = x0* and f(x1*) = (x0* + λ)/2
            target_f_x1 = (x0_star + self.lambda_val) / 2

            # Solve for C and m
            def constraint_eqs(cm_params):
                C, m = cm_params
                f_x0 = self._superhyperbola(x0_star, C, m)
                f_x1 = self._superhyperbola(x1_star, C, m)
                return [f_x0 - x0_star, f_x1 - target_f_x1]

            try:
                # Initial guess for C and m
                m_init = 2.0
                C_init = (x0_star**m_init - self.lambda_val**m_init) ** 2

                cm_result = optimize.least_squares(
                    constraint_eqs,
                    x0=np.array([C_init, m_init]),
                    bounds=([1e-8, 1.0], [np.inf, 10.0]),
                )
                C, m = cm_result.x

                # Evaluate interface equation at both points
                eq1 = self._interface_equation(x0_star, x0_star, x1_star, C, m)
                eq2 = self._interface_equation(x1_star, x0_star, x1_star, C, m)

                return [eq1, eq2]
            except (ValueError, RuntimeError) as e:
                print(f"Optimization failed: {e}")
                return [1e6, 1e6]

        x0_init = 1.1 * self.lambda_val
        x1_init = 1.5 * self.lambda_val

        def interface_residuals(points):
            x0, x1 = points
            eq1, eq2 = system_equations([x0, x1])
            return [eq1, eq2]

        result = optimize.least_squares(
            interface_residuals,
            x0=[x0_init, x1_init],
            bounds=([self.lambda_val, self.lambda_val], [np.inf, np.inf]),
        )
        if result.success:
            x0_star, x1_star = result.x

            # Recalculate C and m
            target_f_x1 = (x0_star + self.lambda_val) / 2

            def constraint_eqs(cm_params):
                C, m = cm_params
                f_x0 = self._superhyperbola(x0_star, C, m)
                f_x1 = self._superhyperbola(x1_star, C, m)
                return [f_x0 - x0_star, f_x1 - target_f_x1]

            m_init = 2.0
            C_init = (x0_star**m_init - self.lambda_val**m_init) ** 2

            cm_result = optimize.least_squares(
                constraint_eqs,
                x0=np.array([C_init, m_init]),
                bounds=([1e-8, 1.0], [np.inf, 10.0]),
            )
            C, m = cm_result.x

            return x0_star, x1_star, C, m
        else:
            raise RuntimeError("Failed to solve interface parameters")

    def solve(self) -> dict:
        """
        Main solve method - computes all solution parameters
        """
        print(f"Solving 2D Stefan problem with beta = {self.beta}, Ti_star = {self.Ti_star}")
        print(f"Neumann parameter lambda = {self.lambda_val:.4f}")

        # Solve for interface parameters
        self.x0_star, self.x1_star, self.C, self.m = self.solve_interface_parameters()

        print(f"Interface parameters: x0_star = {self.x0_star:.4f}, x1_star = {self.x1_star:.4f}")
        print(f"Superhyperbola parameters: C = {self.C:.4f}, m = {self.m:.4f}")

        return {
            "lambda": self.lambda_val,
            "x0_star": self.x0_star,
            "x1_star": self.x1_star,
            "C": self.C,
            "m": self.m,
            "beta": self.beta,
            "Ti_star": self.Ti_star,
        }

    def get_interface_position(self, x_star_array: np.ndarray) -> np.ndarray:
        """
        Get interface position y_star = f(x_star) for given x_star values
        """
        if self.C is None or self.m is None:
            raise RuntimeError("Must call solve() first")

        return np.array([self._superhyperbola(x, self.C, self.m) for x in x_star_array])

    def plot_interface(self, max_x_star: float = 3.0, n_points: int = 100):
        """Plot the interface curve"""
        if self.C is None or self.m is None:
            raise RuntimeError("Must call solve() first")

        x_star = np.linspace(self.x0_star, max_x_star, n_points)
        y_star = self.get_interface_position(x_star)

        plt.figure(figsize=(8, 6))
        plt.plot(x_star, y_star, "b-", linewidth=2, label="Interface")
        plt.plot([0, max_x_star], [0, max_x_star], "k--", alpha=0.5, label=r"$y^* = x^*$")
        plt.axhline(
            y=self.lambda_val,
            color="r",
            linestyle=":",
            label=fr"$\lambda$ = {self.lambda_val:.3f}",
        )
        plt.axvline(x=self.lambda_val, color="r", linestyle=":", alpha=0.7)

        plt.xlabel("x*")
        plt.ylabel("y*")
        plt.title(fr"Interface Position ($\beta={self.beta}$, $T_i^*={self.Ti_star}$)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis("equal")
        plt.show()
