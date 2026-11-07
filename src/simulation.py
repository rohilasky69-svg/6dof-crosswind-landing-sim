import math
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from src.kinematics import (
    quaternion_normalize,
    quaternion_to_dcm,
    quaternion_from_euler,
    dcm_to_euler,
)
from src.aircraft import AircraftModel, Controls, DEG
from src.dynamics import STATE_SIZE, POS, VEL, QUAT, OMEGA, rk4_step, solve_trim
from src.controller import LandingController

MAX_TOUCHDOWN_OFFSET_M = 7.5
MAX_TOUCHDOWN_HEADING_ERROR_DEG = 5.0
MAX_TOUCHDOWN_SINK_MPS = 1.8
MAX_TOUCHDOWN_BANK_DEG = 10.0
MIN_TOUCHDOWN_N_M = 0.0
MAX_TOUCHDOWN_N_M = 800.0
TRIM_RESIDUAL_LIMIT = 1.0e-3
FINAL_SATURATION_WINDOW_S = 3.0
SAFE_MIN_DT = 5.0e-4
SAFE_MAX_DT = 5.0e-2


@dataclass
class SimulationResult:
    wind_east: float
    time: np.ndarray
    states: np.ndarray
    controls: np.ndarray
    phases: List[str]
    derived: Dict[str, np.ndarray]
    metrics: Dict[str, object]


def derived_from_state(states: np.ndarray, model: AircraftModel, wind_ned: np.ndarray) -> Dict[str, np.ndarray]:
    n_rows = len(states)
    result = {
        key: np.zeros(n_rows, dtype=float)
        for key in (
            "north", "east", "down", "altitude", "u_air", "v_air", "w_air",
            "air_speed", "alpha", "beta", "phi", "theta", "psi", "p", "q", "r",
            "ground_speed", "sink_rate",
        )
    }
    for i, state in enumerate(states):
        q = quaternion_normalize(state[QUAT])
        c_bn = quaternion_to_dcm(q)
        ground = c_bn @ state[VEL]
        air_body = c_bn.T @ (ground - wind_ned)
        _, _, air = model.aerodynamic_loads(air_body, state[OMEGA], Controls())
        phi, theta, psi = dcm_to_euler(q)
        result["north"][i], result["east"][i], result["down"][i] = state[POS]
        result["altitude"][i] = -state[POS][2]
        result["u_air"][i], result["v_air"][i], result["w_air"][i] = air_body
        result["air_speed"][i] = air["speed"]
        result["alpha"][i] = air["alpha"]
        result["beta"][i] = air["beta"]
        result["phi"][i], result["theta"][i], result["psi"][i] = phi, theta, psi
        result["p"][i], result["q"][i], result["r"][i] = state[OMEGA]
        result["ground_speed"][i] = np.linalg.norm(ground[:2])
        result["sink_rate"][i] = ground[2]
    return result


def make_initial_state(model: AircraftModel, trim: Dict[str, float], wind_east: float) -> np.ndarray:
    speed = 30.0
    alpha = trim["alpha"]
    wind_ned = np.array([0.0, wind_east, 0.0], dtype=float)
    crab = math.atan2(-wind_east, speed)
    pitch = alpha - 3.0 * DEG
    q = quaternion_from_euler(0.0, pitch, crab)
    c_bn = quaternion_to_dcm(q)
    air_body = np.array([speed * math.cos(alpha), 0.0, speed * math.sin(alpha)])
    ground_ned = c_bn @ air_body + wind_ned
    state = np.zeros(STATE_SIZE, dtype=float)
    state[POS] = np.array([-900.0, 0.0, -48.0], dtype=float)
    state[VEL] = c_bn.T @ ground_ned
    state[QUAT] = q
    state[OMEGA] = 0.0
    return state


def wind_triangle_sanity(model: AircraftModel, trim: Dict[str, float], wind_east: float) -> Dict[str, float]:
    state = make_initial_state(model, trim, wind_east)
    q = quaternion_normalize(state[QUAT])
    c_bn = quaternion_to_dcm(q)
    speed = 30.0
    alpha = trim["alpha"]
    air_body = np.array([speed * math.cos(alpha), 0.0, speed * math.sin(alpha)])
    expected_ground = c_bn @ air_body + np.array([0.0, wind_east, 0.0])
    actual_ground = c_bn @ state[VEL]
    expected_crab = math.atan2(-wind_east, speed) / DEG
    return {
        "wind_triangle_error_mps": float(np.linalg.norm(actual_ground - expected_ground)),
        "wind_triangle_crab_deg": float(expected_crab),
    }


def simulate_landing(
    wind_east: float,
    duration: float,
    dt: float,
    model: Optional[AircraftModel] = None,
    trim: Optional[Dict[str, float]] = None,
) -> SimulationResult:
    if not np.isfinite(dt) or dt < SAFE_MIN_DT or dt > SAFE_MAX_DT:
        raise ValueError(f"dt={dt!r} is outside safe RK4 range [{SAFE_MIN_DT}, {SAFE_MAX_DT}] s")
    if not np.isfinite(duration) or duration <= 0.0:
        raise ValueError("duration must be finite and positive")

    model = model or AircraftModel()
    trim = trim or solve_trim(model)
    wind_ned = np.array([0.0, float(wind_east), 0.0], dtype=float)
    controller = LandingController(model, wind_ned, trim)
    state = make_initial_state(model, trim, wind_east)

    times, states, control_log, saturation_log, phases = [], [], [], [], []
    touchdown_time = None
    failure_reason, ground_contact = "", False
    touchdown_snapshot: Optional[Dict[str, float]] = None
    max_steps = max(2, int(math.ceil(duration / dt)) + 1)

    for step in range(max_steps):
        t = min(step * dt, duration)
        controls, debug = controller.command(state)
        times.append(t)
        states.append(state.copy())
        control_log.append(np.array([controls.aileron, controls.elevator, controls.rudder, controls.throttle]))
        saturation_log.append(np.asarray(debug["saturation_flags"], dtype=bool))
        phases.append(str(debug["phase"]))

        if t >= duration:
            break

        next_state = rk4_step(state, dt, controls, model, wind_ned, ground_contact)
        next_n, next_e, next_down = next_state[POS]

        if not ground_contact and next_down >= 0.0:
            if next_n < -5.0:
                failure_reason = "terrain contact before runway threshold"
                next_state[POS][2] = 0.0
                ground_contact = True
            else:
                controller.touchdown = True
                ground_contact = True
                touchdown_time = t + dt
                contact_derived = derived_from_state(next_state[np.newaxis, :], model, wind_ned)
                touchdown_snapshot = {
                    "north": float(contact_derived["north"][0]),
                    "east": float(contact_derived["east"][0]),
                    "speed": float(contact_derived["ground_speed"][0]),
                    "sink_rate": float(contact_derived["sink_rate"][0]),
                    "heading_error": abs(float(contact_derived["psi"][0])) / DEG,
                    "bank": abs(float(contact_derived["phi"][0])) / DEG,
                }
                next_state[POS][2] = 0.0
                c_bn = quaternion_to_dcm(next_state[QUAT])
                v_ground = c_bn @ next_state[VEL]
                v_ground[2] = 0.0
                next_state[VEL] = c_bn.T @ v_ground
        elif ground_contact:
            next_state[POS][2] = 0.0

        if not np.all(np.isfinite(next_state)):
            failure_reason = "non-finite state encountered"
            break
        if next_n > 1800.0 and not controller.touchdown:
            failure_reason = "passed runway without touchdown"
            break
        if abs(next_e) > 80.0:
            failure_reason = "lateral excursion beyond model runway area"
            break

        state = next_state
        if failure_reason and not controller.touchdown:
            break
        if touchdown_time is not None and t + dt - touchdown_time >= 3.0:
            break

    states_array = np.asarray(states, dtype=float)
    time_array = np.asarray(times, dtype=float)
    controls_array = np.asarray(control_log, dtype=float)
    saturation_array = np.asarray(saturation_log, dtype=bool)
    derived = derived_from_state(states_array, model, wind_ned)

    touched = touchdown_time is not None
    if not failure_reason and not touched:
        failure_reason = "simulation ended before touchdown"

    touchdown_snapshot = touchdown_snapshot or {}
    trim_residual = float(trim.get("residual_norm", math.inf))
    phase_indices = {name: next((i for i, phase in enumerate(phases) if phase == name), None) for name in ("CRAB", "DECRAB/WING_LOW", "FLARE")}
    phase_altitudes = {name: (float(derived["altitude"][i]) if i is not None else math.nan) for name, i in phase_indices.items()}
    phase_checks = (
        phase_indices["CRAB"] is not None
        and phase_indices["DECRAB/WING_LOW"] is not None
        and phase_indices["FLARE"] is not None
        and phase_indices["CRAB"] < phase_indices["DECRAB/WING_LOW"] < phase_indices["FLARE"]
        and 12.0 <= phase_altitudes["DECRAB/WING_LOW"] <= 18.0
        and 6.0 <= phase_altitudes["FLARE"] <= 10.0
    )

    final_window = time_array >= float(touchdown_time) - FINAL_SATURATION_WINDOW_S if touched else np.zeros(len(time_array), dtype=bool)
    final_saturation = saturation_array[final_window] if np.any(final_window) else np.zeros((0, 4), dtype=bool)
    saturation_counts = saturation_array.sum(axis=0) if len(saturation_array) else np.zeros(4, dtype=int)
    final_saturation_counts = final_saturation.sum(axis=0) if len(final_saturation) else np.zeros(4, dtype=int)

    criteria_failures = []
    if trim_residual > TRIM_RESIDUAL_LIMIT:
        criteria_failures.append(f"trim residual {trim_residual:.3g} exceeds {TRIM_RESIDUAL_LIMIT:.3g}")
    if not phase_checks:
        criteria_failures.append("phase transition check failed")
    if touched:
        touchdown_n = touchdown_snapshot.get("north", math.nan)
        touchdown_e = touchdown_snapshot.get("east", math.nan)
        touchdown_sink = max(0.0, touchdown_snapshot.get("sink_rate", math.nan))
        if not (MIN_TOUCHDOWN_N_M <= touchdown_n <= MAX_TOUCHDOWN_N_M):
            criteria_failures.append(f"touchdown N {touchdown_n:.1f} m outside [{MIN_TOUCHDOWN_N_M:.0f}, {MAX_TOUCHDOWN_N_M:.0f}] m")
        if abs(touchdown_e) > MAX_TOUCHDOWN_OFFSET_M:
            criteria_failures.append(f"touchdown E {touchdown_e:.1f} m exceeds +/-{MAX_TOUCHDOWN_OFFSET_M:g} m")
        if touchdown_snapshot.get("heading_error", math.inf) > MAX_TOUCHDOWN_HEADING_ERROR_DEG:
            criteria_failures.append(f"heading error {touchdown_snapshot['heading_error']:.1f} deg exceeds {MAX_TOUCHDOWN_HEADING_ERROR_DEG:g} deg")
        if touchdown_sink > MAX_TOUCHDOWN_SINK_MPS:
            criteria_failures.append(f"sink rate {touchdown_sink:.2f} m/s exceeds {MAX_TOUCHDOWN_SINK_MPS:g} m/s")
        if touchdown_snapshot.get("bank", math.inf) > MAX_TOUCHDOWN_BANK_DEG:
            criteria_failures.append(f"bank {touchdown_snapshot['bank']:.1f} deg exceeds {MAX_TOUCHDOWN_BANK_DEG:g} deg")
        if np.any(final_saturation):
            names = ("aileron", "elevator", "rudder", "throttle")
            saturated_names = [name for name, count in zip(names, final_saturation_counts) if count]
            criteria_failures.append(f"control saturation in final {FINAL_SATURATION_WINDOW_S:g} s: {', '.join(saturated_names)}")

    if failure_reason:
        criteria_failures.insert(0, failure_reason)

    if not touched:
        outcome = "FAILURE"
    elif criteria_failures:
        outcome = "TOUCHDOWN_CRITERIA_FAILURE"
        failure_reason = "; ".join(criteria_failures)
    else:
        outcome = "ACCEPTED_TOUCHDOWN"

    metrics = {
        "outcome": outcome,
        "failure_reason": failure_reason,
        "touchdown": touched,
        "touchdown_time_s": float(touchdown_time) if touchdown_time is not None else math.nan,
        "touchdown_n_m": touchdown_snapshot.get("north", math.nan) if touched else math.nan,
        "touchdown_e_m": touchdown_snapshot.get("east", math.nan) if touched else math.nan,
        "touchdown_speed_mps": touchdown_snapshot.get("speed", math.nan) if touched else math.nan,
        "touchdown_sink_rate_mps": max(0.0, touchdown_snapshot.get("sink_rate", math.nan)) if touched else math.nan,
        "touchdown_heading_error_deg": touchdown_snapshot.get("heading_error", math.nan) if touched else math.nan,
        "touchdown_bank_deg": touchdown_snapshot.get("bank", math.nan) if touched else math.nan,
        "max_bank_deg": float(np.nanmax(np.abs(derived["phi"]))) / DEG,
        "max_sideslip_deg": float(np.nanmax(np.abs(derived["beta"]))) / DEG,
        "samples": len(time_array),
        "criteria_pass": bool(touched and not criteria_failures),
        "criteria_failures": "; ".join(criteria_failures),
        "trim_residual_ok": bool(trim_residual <= TRIM_RESIDUAL_LIMIT),
        "phase_checks_pass": bool(phase_checks),
        "phase_decrab_alt_m": phase_altitudes["DECRAB/WING_LOW"],
        "phase_flare_alt_m": phase_altitudes["FLARE"],
        "wind_triangle_crab_deg": wind_triangle_sanity(model, trim, wind_east)["wind_triangle_crab_deg"],
        "wind_triangle_error_mps": wind_triangle_sanity(model, trim, wind_east)["wind_triangle_error_mps"],
        "saturation_aileron_count": int(saturation_counts[0]),
        "saturation_elevator_count": int(saturation_counts[1]),
        "saturation_rudder_count": int(saturation_counts[2]),
        "saturation_throttle_count": int(saturation_counts[3]),
        "final_saturation_count": int(final_saturation.sum()),
    }
    return SimulationResult(
        wind_east=float(wind_east),
        time=time_array,
        states=states_array,
        controls=controls_array,
        phases=phases,
        derived=derived,
        metrics=metrics,
    )