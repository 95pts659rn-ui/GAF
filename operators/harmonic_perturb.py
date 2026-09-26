import torch
import torch.nn as nn
import numpy as np
import math
from .base import SphericalOperator

class HarmonicPerturbOperator(SphericalOperator):
    def __init__(
        self,
        max_degree,
        amplitude=0.1,
        trainable=True,
        perturb_r=True,
        perturb_theta=True,
        perturb_phi=True
        ):
        super().__init__(trainable=trainable)

        self.max_degree = max_degree
        self.amplitude = amplitude
        self.perturb_r = perturb_r
        self.perturb_theta = perturb_theta
        self.perturb_phi = perturb_phi

        num_coeffs = sum(2 * l + 1 for l in range(max_degree + 1))

        if trainable:
            if perturb_r:
                self.r_coeffs = nn.Parameter(
                    torch.randn(num_coeffs) * amplitude * 0.1
                )

            if perturb_theta:
                self.theta_coeffs = nn.Parameter(
                    torch.randn(num_coeffs) * amplitude * 0.1
                )

            if perturb_phi:
                self.phi_coeffs = nn.Parameter(
                    torch.randn(num_coeffs) * amplitude * 0.1
                )
        else:
            if perturb_r:
                self.register_buffer(
                    'r_coeffs', 
                    torch.randn(num_coeffs) * amplitude * 0.1
                )

            if perturb_theta:
                self.register_buffer(
                    'theta_coeffs', 
                    torch.randn(num_coeffs) * amplitude * 0.1
                )

            if perturb_phi:
                self.register_buffer(
                    'phi_coeffs', 
                    torch.randn(num_coeffs) * amplitude * 0.1
                )

        self.lm_pairs = []
        for l in range(max_degree + 1):
            for m in range(-l, l + 1):
                self.lm_pairs.append((l, m))

    def _evaluate_spherical_harmonics(
        self, 
        theta, 
        phi):
        batch_size = theta.shape[0]
        harmonics = []

        theta = torch.clamp(theta, min=1e-6, max=np.pi - 1e-6)

        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)

        for l, m in self.lm_pairs:
            if m == 0:
                if l == 0:
                    P_l = torch.ones_like(cos_theta)
                elif l == 1:
                    P_l = cos_theta
                else:
                    P_l_minus_2 = torch.ones_like(cos_theta)  # P_0
                    P_l_minus_1 = cos_theta  # P_1

                    for ll in range(2, l + 1):
                        P_l = ((2 * ll - 1) * cos_theta * P_l_minus_1 - 
                              (ll - 1) * P_l_minus_2) / ll
                        P_l_minus_2 = P_l_minus_1
                        P_l_minus_1 = P_l

                norm_factor = torch.sqrt(torch.tensor((2 * l + 1) / (4 * np.pi), dtype=P_l.dtype, device=P_l.device))
                harmonic = norm_factor * P_l

            else:
                abs_m = abs(m)

                P_mm = torch.ones_like(cos_theta)

                for j in range(1, abs_m + 1):
                    P_mm = P_mm * (-1) * (2 * j - 1) * sin_theta

                if l == abs_m:
                    P_lm = P_mm
                else:
                    P_mm_plus_1 = cos_theta * (2 * abs_m + 1) * P_mm

                    if l == abs_m + 1:
                        P_lm = P_mm_plus_1
                    else:
                        P_l_minus_2 = P_mm
                        P_l_minus_1 = P_mm_plus_1

                        for ll in range(abs_m + 2, l + 1):
                            P_lm = ((2 * ll - 1) * cos_theta * P_l_minus_1 - 
                                  (ll + abs_m - 1) * P_l_minus_2) / (ll - abs_m)
                            P_l_minus_2 = P_l_minus_1
                            P_l_minus_1 = P_lm

                norm_factor = torch.sqrt(
                    torch.tensor((2 * l + 1) * math.factorial(l - abs_m) / 
                    (4 * np.pi * math.factorial(l + abs_m)), dtype=P_lm.dtype, device=P_lm.device)
                )

                if m > 0:
                    harmonic = norm_factor * P_lm * torch.cos(m * phi)
                else:  # m < 0
                    harmonic = norm_factor * P_lm * torch.sin(abs_m * phi)

            harmonics.append(harmonic)

        return harmonics

    def forward(self, coordinates):
        r = coordinates[..., 0:1]  # Keep dimension for easier broadcasting
        theta = coordinates[..., 1:2]
        phi = coordinates[..., 2:3]

        harmonics = self._evaluate_spherical_harmonics(theta, phi)

        harmonics_tensor = torch.cat(harmonics, dim=-1)

        r_new = r
        theta_new = theta
        phi_new = phi

        if self.perturb_r:
            r_perturbation = torch.matmul(harmonics_tensor, self.r_coeffs.unsqueeze(-1))
            r_new = r + self.amplitude * r * r_perturbation
            r_new = torch.clamp(r_new, min=1e-8)

        if self.perturb_theta:
            theta_perturbation = torch.matmul(harmonics_tensor, self.theta_coeffs.unsqueeze(-1))
            theta_new = theta + self.amplitude * theta_perturbation
            theta_new = torch.clamp(theta_new, min=1e-8, max=np.pi - 1e-8)

        if self.perturb_phi:
            phi_perturbation = torch.matmul(harmonics_tensor, self.phi_coeffs.unsqueeze(-1))
            phi_new = phi + self.amplitude * phi_perturbation
            phi_new = torch.remainder(phi_new, 2 * np.pi)

        result = torch.cat([r_new, theta_new, phi_new], dim=-1)

        return result
