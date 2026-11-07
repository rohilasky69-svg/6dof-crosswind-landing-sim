import math
from typing import Dict
import numpy as np

try:
    from scipy.optimize import least_squares
except ImportError:
    least_squares = None

from src.kinematics import quaternion_normalize, quaternion_to_dcm, quaternion_from_euler, quaternion_rate
from src.aircraft import AircraftModel, Controls, DEG

STATE_SIZE = 13
POS = slice(0, 3)
VEL = slice(3, 6)
QUAT = slice(6, 10)
OMEGA = slice(10, 13)

def rhs(state: np.ndarray, controls: Controls, model: AircraftModel, wind_ned: np.ndarray, ground_contact: bool = False) -> np.ndarray:
    pos = state[POS]
    vel_body_ground = state[VEL]
    q = quaternion_normalize(state[QUAT])
    omega = state[OMEGA]
    c_bn = quaternion_to_dcm(q)
    ground_velocity_ned = c_bn @ vel_body_ground
    air_velocity_body = c_bn.T @ (ground_velocity_ned - wind_ned)
    force_body, moment_body, _ = model.aerodynamic_loads(air_velocity_body, omega, controls)
    gravity_ned = np.array([0.0, 0.0, model.gravity], dtype=float)
    force_body = force_body + model.mass * (c_bn.T @ gravity_ned)
    if ground_contact:
        v_ned = ground_velocity_ned.copy()
        horizontal = v_ned[:2]
        hspeed = float(np.linalg.norm(horizontal))
        if hspeed > 1.0e-6:
            friction_ned = np.array([horizontal[0], horizontal[1], 0.0], dtype=float)
            force_body += c_bn.T @ (-0.025 * model.mass * model.gravity * friction_ned / hspeed)
        normal_needed = max(0.0, -(c_bn @ force_body)[2])
        force_body += c_bn.T @ np.array([0.0, 0.0, normal_needed], dtype=float)
    vel_dot_body = force_body / model.mass - np.cross(omega, vel_body_ground)
    inertia = model.inertia
    omega_dot = np.linalg.solve(inertia, moment_body - np.cross(omega, inertia @ omega))
    pos_dot_ned = ground_velocity_ned
    q_dot = quaternion_rate(q, omega)
    derivative = np.zeros(STATE_SIZE, dtype=float)
    derivative[POS] = pos_dot_ned
    derivative[VEL] = vel_dot_body
    derivative[QUAT] = q_dot
    derivative[OMEGA] = omega_dot
    return derivative

def rk4_step(state: np.ndarray, dt: float, controls: Controls, model: AircraftModel, wind_ned: np.ndarray, ground_contact: bool) -> np.ndarray:
    k1 = rhs(state, controls, model, wind_ned, ground_contact)
    k2 = rhs(state + 0.5 * dt * k1, controls, model, wind_ned, ground_contact)
    k3 = rhs(state + 0.5 * dt * k2, controls, model, wind_ned, ground_contact)
    k4 = rhs(state + dt * k3, controls, model, wind_ned, ground_contact)
    result = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    result[QUAT] = quaternion_normalize(result[QUAT])
    return result

def solve_trim(model: AircraftModel, speed: float = 30.0) -> Dict[str, float]:
    def residual(x: np.ndarray) -> np.ndarray:
        alpha, elevator, throttle = x
        q = quaternion_from_euler(0.0, alpha, 0.0)
        c_bn = quaternion_to_dcm(q)
        air_body = np.array([speed * math.cos(alpha), 0.0, speed * math.sin(alpha)])
        force, moment, _ = model.aerodynamic_loads(air_body, np.zeros(3), Controls(elevator=elevator, throttle=throttle))
        total_force = force + model.mass * (c_bn.T @ np.array([0.0, 0.0, model.gravity]))
        return np.array([total_force[0] / model.weight, total_force[2] / model.weight, moment[1] / (model.weight * model.mean_chord)], dtype=float)
    initial = np.array([8.0 * DEG, -2.0 * DEG, 0.35], dtype=float)
    lower = np.array([-2.0 * DEG, -25.0 * DEG, 0.0], dtype=float)
    upper = np.array([25.0 * DEG, 25.0 * DEG, 1.0], dtype=float)
    if least_squares is not None:
        solution = least_squares(residual, initial, bounds=(lower, upper), max_nfev=200)
        x = solution.x
    else:
        x = initial
    r = residual(x)
    return {"alpha": float(x[0]), "elevator": float(x[1]), "throttle": float(x[2]), "residual_norm": float(np.linalg.norm(r)), "solver": "scipy" if least_squares else "guess"}