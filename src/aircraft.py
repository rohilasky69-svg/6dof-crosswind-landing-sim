import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
from src.kinematics import clamp

DEG = math.pi / 180.0

@dataclass
class Controls:
    aileron: float = 0.0
    elevator: float = 0.0
    rudder: float = 0.0
    throttle: float = 0.0

    def array(self) -> np.ndarray:
        return np.array([self.aileron, self.elevator, self.rudder, self.throttle], dtype=float)

    def limited(self) -> "Controls":
        return Controls(
            aileron=clamp(self.aileron, -25.0 * DEG, 25.0 * DEG),
            elevator=clamp(self.elevator, -25.0 * DEG, 25.0 * DEG),
            rudder=clamp(self.rudder, -30.0 * DEG, 30.0 * DEG),
            throttle=clamp(self.throttle, 0.0, 1.0),
        )

    def saturation_flags(self) -> np.ndarray:
        raw = self.array()
        limits = np.array([25.0 * DEG, 25.0 * DEG, 30.0 * DEG, 1.0], dtype=float)
        lower = np.array([-25.0 * DEG, -25.0 * DEG, -30.0 * DEG, 0.0], dtype=float)
        return (raw < lower - 1.0e-10) | (raw > limits + 1.0e-10)

@dataclass
class AircraftModel:
    mass: float = 1000.0
    wing_area: float = 16.2
    wing_span: float = 10.9
    mean_chord: float = 1.50
    rho: float = 1.225
    gravity: float = 9.80665
    ix: float = 1250.0
    iy: float = 1800.0
    iz: float = 3000.0
    ixz: float = 0.0
    cl0: float = 0.43
    cl_alpha: float = 4.80
    cl_elevator: float = 0.35
    cd0: float = 0.035
    induced_drag: float = 0.052
    cy_beta: float = -0.72
    cy_aileron: float = 0.08
    cy_rudder: float = 0.25
    cl_beta: float = -0.12
    cl_p: float = -0.50
    cl_r: float = 0.20
    cl_aileron: float = 0.18
    cl_rudder: float = 0.04
    cm0: float = 0.065
    cm_alpha: float = -0.95
    cm_q: float = -7.0
    cm_elevator: float = -1.10
    cn_beta: float = 0.25
    cn_p: float = -0.06
    cn_r: float = -0.22
    cn_aileron: float = 0.05
    cn_rudder: float = 0.12
    max_thrust: float = 2600.0

    @property
    def weight(self) -> float: return self.mass * self.gravity
    @property
    def inertia(self) -> np.ndarray: return np.diag([self.ix, self.iy, self.iz])

    def aerodynamic_loads(self, air_velocity_body: np.ndarray, omega_body: np.ndarray, controls: Controls) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
        u, v, w = np.asarray(air_velocity_body, dtype=float)
        speed = max(float(np.linalg.norm(air_velocity_body)), 1.0)
        alpha = math.atan2(w, max(abs(u), 0.5))
        beta = math.asin(clamp(v / speed, -1.0, 1.0))
        alpha_eff = clamp(alpha, -25.0 * DEG, 25.0 * DEG)
        q_dyn = 0.5 * self.rho * speed * speed
        pbar = omega_body[0] * self.wing_span / (2.0 * speed)
        qbar = omega_body[1] * self.mean_chord / (2.0 * speed)
        rbar = omega_body[2] * self.wing_span / (2.0 * speed)
        ctl = controls.limited()
        cl = clamp(self.cl0 + self.cl_alpha * alpha_eff + self.cl_elevator * ctl.elevator, -1.6, 1.8)
        cd = self.cd0 + self.induced_drag * cl * cl
        cy = self.cy_beta * beta + self.cy_aileron * ctl.aileron + self.cy_rudder * ctl.rudder
        roll_coeff = (self.cl_beta * beta + self.cl_p * pbar + self.cl_r * rbar + self.cl_aileron * ctl.aileron + self.cl_rudder * ctl.rudder)
        pitch_coeff = (self.cm0 + self.cm_alpha * alpha_eff + self.cm_q * qbar + self.cm_elevator * ctl.elevator)
        yaw_coeff = (self.cn_beta * beta + self.cn_p * pbar + self.cn_r * rbar + self.cn_aileron * ctl.aileron + self.cn_rudder * ctl.rudder)
        lift = q_dyn * self.wing_area * cl
        drag = q_dyn * self.wing_area * cd
        side = q_dyn * self.wing_area * cy
        ca, sa = math.cos(alpha), math.sin(alpha)
        force_aero = np.array([-drag * ca + lift * sa, side, -drag * sa - lift * ca], dtype=float)
        moment_aero = q_dyn * self.wing_area * np.array([self.wing_span * roll_coeff, self.mean_chord * pitch_coeff, self.wing_span * yaw_coeff], dtype=float)
        force_prop = np.array([self.max_thrust * ctl.throttle, 0.0, 0.0], dtype=float)
        data = {"speed": speed, "alpha": alpha, "beta": beta, "q_dyn": q_dyn, "cl": cl, "cd": cd, "cy": cy, "lift": lift, "drag": drag}
        return force_aero + force_prop, moment_aero, data