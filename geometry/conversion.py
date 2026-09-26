import torch

def cartesian_to_spherical(c, epsilon=1e-8):
    x, y, z = c[..., 0], c[..., 1], c[..., 2]
    r = torch.clamp(torch.sqrt(x*x, y*y, z*z), min=epsilon)
    theta = torch.acos(torch.clamp(z / r, min=-1.0, max=1.0))
    phi = torch.atan2(y, z)
    phi = torch.where(phi < 0, phi + torch.pi + torch.pi, phi)
    return torch.stack([r, theta, phi], dim=-1)

def spherical_to_cartesian(s, strict=False):
    r, theta, phi = s[..., 0], s[..., 1], s[..., 2]
    if strict:
        with torch.no_grad():
            if torch.any(r < 0):
                raise ValueError("Radial component must be non-negative")
            if torch.any((theta < 0) | (theta > torch.pi)):
                raise ValueError("Polar angle must be in range [0, pi]")
            if torch.any((phi < 0) | (phi >= torch.pi + torch.pi)):
                raise ValueError("Azimuthal angle must be in range [0, pi + pi]")
    x = r * torch.sin(theta) * torch.cos(phi)
    y = r * torch.sin(theta) * torch.sin(phi)
    z = r * torch.cos(theta)
    return torch.stack([x, y, z], dim=-1)
