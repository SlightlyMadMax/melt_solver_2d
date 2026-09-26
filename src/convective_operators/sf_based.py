import numpy as np

from src.convective_operators.base_convective_operator import (
    BaseConvectiveOperator,
    ConvectiveTermForm,
)
from src.parameters.config import ExperimentConfig


class StreamFunctionBasedConvectiveOperator(BaseConvectiveOperator):
    def __init__(self, form: ConvectiveTermForm, cfg: ExperimentConfig):
        super().__init__(cfg=cfg)
        self.form = form
        n_y, n_x = self.cfg.geometry.n_y, self.cfg.geometry.n_x
        self._v_x: np.ndarray = np.zeros((n_y, n_x))
        self._v_y: np.ndarray = np.zeros((n_y, n_x))

    def __call__(
        self,
        conv_x,
        conv_y,
        sf: np.ndarray,
        recalculate_velocity: bool = True,
        correction_x: np.ndarray | None = None,
        correction_y: np.ndarray | None = None,
        convected_quantity: np.ndarray | None = None,
    ) -> None:
        if recalculate_velocity:
            self.compute_velocity_from_sf(sf=sf)

        if self.form == ConvectiveTermForm.UPWIND_FC:
            self._compute_upwind_components_at_faces(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.UPWIND_NC:
            self._compute_upwind_components_at_nodes(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.DIVERGENT_CENTRAL:
            self._compute_div_components(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.NON_DIVERGENT_CENTRAL:
            self._compute_non_div_components(result_x=conv_x, result_y=conv_y)
        elif self.form == ConvectiveTermForm.SYMMETRIC:
            tmp_x, tmp_y = np.zeros_like(conv_x), np.zeros_like(conv_y)
            self._compute_div_components(result_x=tmp_x, result_y=tmp_y)
            self._compute_non_div_components(result_x=conv_x, result_y=conv_y)
            conv_x[:] = 0.5 * (tmp_x + conv_x)
            conv_y[:] = 0.5 * (tmp_y + conv_y)
        elif self.form == ConvectiveTermForm.DEFERRED_CORRECTION:
            self._compute_upwind_components_at_nodes(result_x=conv_x, result_y=conv_y)
            self._compute_correction(
                result_x=correction_x,
                result_y=correction_y,
                convected_quantity=convected_quantity,
            )
        elif self.form == ConvectiveTermForm.DEFERRED_CORRECTION_DIV:
            self._compute_rusanov_div_components(result_x=conv_x, result_y=conv_y)
            self._compute_div_correction(
                result_x=correction_x,
                result_y=correction_y,
                convected_quantity=convected_quantity,
            )
        else:
            raise NotImplementedError(f"ConvectiveTermForm {self.form} not supported")

    def compute_velocity_from_sf(self, sf: np.ndarray) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        v_x[1:-1, 1:-1] = inv_2dy * (sf[2:, 1:-1] - sf[:-2, 1:-1])
        v_y[1:-1, 1:-1] = -inv_2dx * (sf[1:-1, 2:] - sf[1:-1, :-2])

    def _compute_upwind_components_at_faces(
        self, result_x: np.ndarray, result_y: np.ndarray
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        jm, jp = slice(1, -1), slice(2, None)
        im, ip = slice(1, -1), slice(2, None)
        im2, jm2 = slice(None, -2), slice(None, -2)

        v_x_c = v_x[jm, im]
        v_x_r = v_x[jm, ip]
        v_x_l = v_x[jm, im2]
        v_y_c = v_y[jm, im]
        v_y_d = v_y[jp, im]
        v_y_u = v_y[jm2, im]

        v_x_p = 0.5 * (v_x_c + v_x_r)
        v_x_m = 0.5 * (v_x_c + v_x_l)
        v_y_p = 0.5 * (v_y_c + v_y_d)
        v_y_m = 0.5 * (v_y_c + v_y_u)

        result_x[jm, im, 0] = inv_2dx * (v_x_p - np.abs(v_x_p))
        result_x[jm, im, 1] = inv_2dx * (
            (v_x_p + np.abs(v_x_p)) - (v_x_m - np.abs(v_x_m))
        )
        result_x[jm, im, 2] = -inv_2dx * (v_x_m + np.abs(v_x_m))

        result_y[jm, im, 0] = inv_2dy * (v_y_p - np.abs(v_y_p))
        result_y[jm, im, 1] = inv_2dy * (
            (v_y_p + np.abs(v_y_p)) - (v_y_m - np.abs(v_y_m))
        )
        result_y[jm, im, 2] = -inv_2dy * (v_y_m + np.abs(v_y_m))

    def _compute_upwind_components_at_nodes(
        self, result_x: np.ndarray, result_y: np.ndarray
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_dx = 1.0 / dx
        inv_dy = 1.0 / dy

        jm, im = slice(1, -1), slice(1, -1)

        v_x_p = np.maximum(v_x[jm, im], 0.0)
        v_x_m = np.minimum(v_x[jm, im], 0.0)

        result_x[jm, im, 0] = v_x_m * inv_dx  # phi[i+1]
        result_x[jm, im, 1] = (v_x_p - v_x_m) * inv_dx  # phi[i]
        result_x[jm, im, 2] = -v_x_p * inv_dx  # phi[i-1]

        v_y_p = np.maximum(v_y[jm, im], 0.0)
        v_y_m = np.minimum(v_y[jm, im], 0.0)

        result_y[jm, im, 0] = v_y_m * inv_dy  # phi[j+1, i]
        result_y[jm, im, 1] = (v_y_p - v_y_m) * inv_dy  # phi[j, i]
        result_y[jm, im, 2] = -v_y_p * inv_dy  # phi[j-1, i]

    def _compute_correction(
        self, result_x: np.ndarray, result_y: np.ndarray, convected_quantity: np.ndarray
    ):
        q = convected_quantity
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)
        eps = 1e-10

        # interior slices corresponding to loop indices: i,j in range(2, n_x-2) / (2, n_y-2)
        cs_x = slice(2, -2)  # columns 2..n_x-3
        cs_y = slice(2, -2)  # rows 2..n_y-3

        # center and neighbor stencils (all have shape (n_y-4, n_x-4))
        C = q[cs_y, cs_x]  # q[j, i]
        L1 = q[cs_y, 1:-3]  # q[j, i-1]
        L2 = q[cs_y, 0:-4]  # q[j, i-2]
        R1 = q[cs_y, 3:-1]  # q[j, i+1]
        R2 = q[cs_y, 4:]  # q[j, i+2]

        U1 = q[1:-3, cs_x]  # q[j-1, i]
        U2 = q[0:-4, cs_x]  # q[j-2, i]
        D1 = q[3:-1, cs_x]  # q[j+1, i]
        D2 = q[4:, cs_x]  # q[j+2, i]

        # corresponding velocity blocks
        Vx = v_x[cs_y, cs_x]
        Vy = v_y[cs_y, cs_x]

        # --- X-direction correction ---

        # branch where v_x > 0
        pos_mask_x = Vx > 0

        # second-order finite-difference piece for positive velocity
        corr_x_pos = Vx * (C - 2.0 * L1 + L2) * inv_2dx
        num_x_pos = C - L1
        den_x_pos = R1 - C
        r_x_pos = num_x_pos / (den_x_pos + eps)
        limiter_x_pos = np.maximum(0, np.minimum(1, r_x_pos))
        corr_x_pos *= limiter_x_pos

        # branch where v_x <= 0  (original code uses else i.e. v_x <=0)
        corr_x_neg = -Vx * (R2 - 2.0 * R1 + C) * inv_2dx
        num_x_neg = R1 - C
        den_x_neg = C - L1
        r_x_neg = num_x_neg / (den_x_neg + eps)
        limiter_x_neg = np.maximum(0, np.minimum(1, r_x_neg))
        corr_x_neg *= limiter_x_neg

        # pick per-element which branch to use
        corr_x = np.where(pos_mask_x, corr_x_pos, corr_x_neg)

        # write into result_x interior
        result_x[cs_y, cs_x] = corr_x

        # --- Y-direction correction ---

        pos_mask_y = Vy > 0

        # vy > 0 branch
        corr_y_pos = Vy * (C - 2.0 * U1 + U2) * inv_2dy
        num_y_pos = C - U1
        den_y_pos = D1 - C
        r_y_pos = num_y_pos / (den_y_pos + eps)
        limiter_y_pos = np.maximum(0, np.minimum(1, r_y_pos))
        corr_y_pos *= limiter_y_pos

        # vy <= 0 branch
        corr_y_neg = -Vy * (D2 - 2.0 * D1 + C) * inv_2dy
        num_y_neg = D1 - C
        den_y_neg = C - U1
        r_y_neg = num_y_neg / (den_y_neg + eps)
        limiter_y_neg = np.maximum(0, np.minimum(1, r_y_neg))
        corr_y_neg *= limiter_y_neg

        corr_y = np.where(pos_mask_y, corr_y_pos, corr_y_neg)

        # write into result_y interior
        result_y[cs_y, cs_x] = corr_y

    def _compute_div_components(
        self, result_x: np.ndarray, result_y: np.ndarray
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        jm, im = slice(1, -1), slice(1, -1)
        ip, im2 = slice(2, None), slice(None, -2)
        jp, jm2 = slice(2, None), slice(None, -2)

        result_x[jm, im, 0] = inv_2dx * v_x[jm, ip]  # v_x[j, i+1]
        result_x[jm, im, 1] = 0.0
        result_x[jm, im, 2] = -inv_2dx * v_x[jm, im2]  # -v_x[j, i-1]

        result_y[jm, im, 0] = inv_2dy * v_y[jp, im]  # v_y[j+1, i]
        result_y[jm, im, 1] = 0.0
        result_y[jm, im, 2] = -inv_2dy * v_y[jm2, im]  # -v_y[j-1, i]

    @staticmethod
    def _face_dissipation(v: np.ndarray, axis: int) -> np.ndarray:
        """
        Upwind dissipation coefficient max(|v_i|, |v_i+1|) on the faces along `axis`.

        Faces that touch a wall node carry none: the wall velocity is zero, so the face
        stays exactly the central one, which keeps the flux through the wall zero.
        """
        speed = np.abs(v)
        if axis == 1:
            alpha = np.maximum(speed[:, :-1], speed[:, 1:])
            alpha[:, 0] = 0.0
            alpha[:, -1] = 0.0
        else:
            alpha = np.maximum(speed[:-1, :], speed[1:, :])
            alpha[0, :] = 0.0
            alpha[-1, :] = 0.0
        return alpha

    def _compute_rusanov_div_components(
        self, result_x: np.ndarray, result_y: np.ndarray
    ) -> None:
        """
        Implicit part of the divergent deferred correction: d(v phi)/dx with face fluxes

            F_i+1/2 = (v_i phi_i + v_i+1 phi_i+1) / 2 - alpha_i+1/2 (phi_i+1 - phi_i) / 2,

        alpha = max(|v_i|, |v_i+1|). The first term is exactly the divergent central
        flux, so the fluxes telescope over the domain; the second is the upwind
        dissipation, which also keeps every off-diagonal coefficient non-positive.
        """
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        jm, im = slice(1, -1), slice(1, -1)

        a_x = self._face_dissipation(v_x, axis=1)  # (n_y, n_x - 1)
        a_e = a_x[jm, 1:]  # face i+1/2 of node i, i = 1..n_x-2
        a_w = a_x[jm, :-1]  # face i-1/2
        half_inv_dx = 0.5 / dx
        result_x[jm, im, 0] = half_inv_dx * (v_x[jm, 2:] - a_e)
        result_x[jm, im, 1] = half_inv_dx * (a_e + a_w)
        result_x[jm, im, 2] = -half_inv_dx * (v_x[jm, :-2] + a_w)

        a_y = self._face_dissipation(v_y, axis=0)  # (n_y - 1, n_x)
        a_n = a_y[1:, im]  # face j+1/2 of node j
        a_s = a_y[:-1, im]  # face j-1/2
        half_inv_dy = 0.5 / dy
        result_y[jm, im, 0] = half_inv_dy * (v_y[2:, im] - a_n)
        result_y[jm, im, 1] = half_inv_dy * (a_n + a_s)
        result_y[jm, im, 2] = -half_inv_dy * (v_y[:-2, im] + a_s)

    @staticmethod
    def _limited_face_correction(
        q: np.ndarray, v: np.ndarray, alpha: np.ndarray, axis: int
    ) -> np.ndarray:
        """
        Face correction alpha * psi(r) * (phi_i+1 - phi_i) / 2 that takes the upwind flux
        back towards the central one; psi = 1 recovers it exactly.

        psi is the minmod limiter of r, the ratio of the jump across the next face
        upstream to the jump across this one. Where that face does not exist, psi = 0.
        """
        if axis == 0:
            q, v, alpha = q.T, v.T, alpha.T

        jump = q[:, 1:] - q[:, :-1]  # across face i+1/2
        upstream = np.zeros_like(jump)
        upstream[:, 1:] = jump[:, :-1]  # phi_i - phi_i-1, used when the flow is +
        downstream = np.zeros_like(jump)
        downstream[:, :-1] = jump[:, 1:]  # phi_i+2 - phi_i+1, used when it is -

        face_v = 0.5 * (v[:, :-1] + v[:, 1:])
        upwind_jump = np.where(face_v >= 0.0, upstream, downstream)
        # r = upwind_jump / jump, set to 0 where the face jump vanishes
        r = upwind_jump * jump / (jump * jump + 1e-300)
        psi = np.clip(r, 0.0, 1.0)
        corr = 0.5 * alpha * psi * jump

        return corr.T if axis == 0 else corr

    def _compute_div_correction(
        self, result_x: np.ndarray, result_y: np.ndarray, convected_quantity: np.ndarray
    ) -> None:
        """
        Explicit part of the divergent deferred correction, as a difference of face
        corrections so that it telescopes over the domain like the fluxes themselves.
        """
        q = convected_quantity
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        jm, im = slice(1, -1), slice(1, -1)

        c_x = self._limited_face_correction(
            q, v_x, self._face_dissipation(v_x, axis=1), axis=1
        )
        result_x[:, :] = 0.0
        result_x[jm, im] = (c_x[jm, 1:] - c_x[jm, :-1]) / dx

        c_y = self._limited_face_correction(
            q, v_y, self._face_dissipation(v_y, axis=0), axis=0
        )
        result_y[:, :] = 0.0
        result_y[jm, im] = (c_y[1:, im] - c_y[:-1, im]) / dy

    def _compute_non_div_components(
        self, result_x: np.ndarray, result_y: np.ndarray
    ) -> None:
        dx, dy, _ = self.cfg.scaled_grid_steps
        v_x, v_y = self._v_x, self._v_y
        inv_2dx = 1.0 / (2.0 * dx)
        inv_2dy = 1.0 / (2.0 * dy)

        interior = (slice(1, -1), slice(1, -1))

        result_x[interior + (0,)] = inv_2dx * v_x[interior]
        result_x[interior + (1,)] = 0.0
        result_x[interior + (2,)] = -inv_2dx * v_x[interior]

        result_y[interior + (0,)] = inv_2dy * v_y[interior]
        result_y[interior + (1,)] = 0.0
        result_y[interior + (2,)] = -inv_2dy * v_y[interior]
