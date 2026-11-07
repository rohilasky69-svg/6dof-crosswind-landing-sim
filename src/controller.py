import math
from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np

from src.kinematics import clamp, wrap_angle, quaternion_normalize, quaternion_to_dcm, dcm_to_euler
from src.aircraft import AircraftModel, Controls, DEG
from src.dynamics import POS, VEL, QUAT, OMEGA

@dataclass
class LandingController:
    model: AircraftModel
    wind_ned: np.ndarray
    trim: Dict[str, float]
    runway_heading: float = 0.0
    touchdown: bool = False
    phase: str = "CRAB"

    def choose_phase(self, state: np.ndarray) -> str:
        if self.touchdown: return "TOUCHDOWN"
        altitude = -float(state[POS][2])
        if altitude <= 8.0: return "FLARE"
        if altitude <= 15.0: return "DECRAB/WING_LOW"
        return "CRAB"

    def command(self, state: np.ndarray) -> Tuple[Controls, Dict[str, float]]:
        self.phase = self.choose_phase(state)
        q = quaternion_normalize(state[QUAT])
        phi, theta, psi = dcm_to_euler(q)
        omega = state[OMEGA]
        c_bn = quaternion_to_dcm(q)
        ground_ned = c_bn @ state[VEL]
        air_body = c_bn.T @ (ground_ned - self.wind_ned)
        _, _, air = self.model.aerodynamic_loads(air_body, omega, Controls())
        speed = air["speed"]
        _, east, down = state[POS]
        altitude = -float(down)
        wind_east = float(self.wind_ned[1])
        crab_angle = math.atan2(-wind_east, max(28.0, speed))
        if self.phase == "CRAB":
            target_yaw = self.runway_heading + crab_angle
            target_beta = 0.0
            target_speed = 30.0
        elif self.phase == "DECRAB/WING_LOW":
            target_yaw = self.runway_heading
            target_beta = math.atan2(-wind_east, max(28.0, speed))
            target_speed = 28.0
        elif self.phase == "FLARE":
            target_yaw = self.runway_heading
            target_beta = math.atan2(-wind_east, max(24.0, speed))
            target_speed = 25.0
        else:
            target_yaw = self.runway_heading
            target_beta = math.atan2(-wind_east, max(22.0, speed))
            target_speed = 22.0
        yaw_error = wrap_angle(target_yaw - psi)
        wind_bank = -0.050 * wind_east if self.phase != "CRAB" else 0.0
        phi_cmd = clamp(0.80 * yaw_error - 0.018 * float(east) + wind_bank, -12.0 * DEG, 12.0 * DEG)
        if self.phase in ("CRAB", "DECRAB/WING_LOW"):
            target_altitude = max(0.0, -float(state[POS][0]) * math.tan(3.0 * DEG) - 5.5)
            gamma_correction = clamp(0.020 * (target_altitude - altitude), -8.0 * DEG, 4.0 * DEG)
            theta_cmd = self.trim["alpha"] - 3.0 * DEG + gamma_correction
        elif self.phase == "FLARE":
            theta_cmd = self.trim["alpha"] + 0.5 * DEG
        else:
            theta_cmd = self.trim["alpha"] + 1.5 * DEG
        elevator = (self.trim["elevator"] - 2.50 * (theta_cmd - theta) + 0.45 * omega[1] + 0.010 * (target_speed - speed))
        aileron = 1.55 * (phi_cmd - phi) - 0.50 * omega[0]
        rudder = 0.35 * (air["beta"] - target_beta) + 3.00 * yaw_error - 0.15 * omega[2]
        if self.phase == "TOUCHDOWN":
            throttle = 0.0
            elevator = self.trim["elevator"] - 0.45 * (theta - self.trim["alpha"])
        elif self.phase == "FLARE":
            throttle = 0.15 + 0.025 * (target_speed - speed)
        else:
            throttle = self.trim["throttle"] + 0.035 * (target_speed - speed)
        raw_controls = Controls(aileron, elevator, rudder, throttle)
        controls = raw_controls.limited()
        debug = {"phase": self.phase, "speed": speed, "alpha": air["alpha"], "beta": air["beta"], "phi_cmd": phi_cmd, "theta_cmd": theta_cmd, "target_yaw": target_yaw, "target_beta": target_beta, "raw_controls": raw_controls.array(), "saturation_flags": raw_controls.saturation_flags()}
        return controls, debug