import math
import numpy as np
from typing import Tuple

def clamp(value: float, lower: float, upper: float) -> float:
    return float(np.clip(value, lower, upper))

def wrap_angle(angle: float) -> float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi

def quaternion_normalize(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if not np.isfinite(norm) or norm < 1.0e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return np.asarray(q, dtype=float) / norm

def quaternion_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ], dtype=float)

def quaternion_from_euler(phi: float, theta: float, psi: float) -> np.ndarray:
    cr, sr = math.cos(phi / 2.0), math.sin(phi / 2.0)
    cp, sp = math.cos(theta / 2.0), math.sin(theta / 2.0)
    cy, sy = math.cos(psi / 2.0), math.sin(psi / 2.0)
    return quaternion_normalize(np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=float))

def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quaternion_normalize(q)
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=float)

def dcm_to_euler(q: np.ndarray) -> Tuple[float, float, float]:
    c = quaternion_to_dcm(q)
    phi = math.atan2(c[2, 1], c[2, 2])
    theta = math.asin(clamp(-c[2, 0], -1.0, 1.0))
    psi = math.atan2(c[1, 0], c[0, 0])
    return phi, theta, psi

def quaternion_rate(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    return 0.5 * quaternion_multiply(q, np.array([0.0, *omega_body], dtype=float))