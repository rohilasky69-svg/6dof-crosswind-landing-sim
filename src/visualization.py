import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Polygon, Rectangle

from src.aircraft import AircraftModel, DEG
from src.simulation import SimulationResult, MAX_TOUCHDOWN_OFFSET_M, MAX_TOUCHDOWN_SINK_MPS, MAX_TOUCHDOWN_BANK_DEG

C_BG, C_PANEL, C_GRID = '#0b0f14', '#111823', '#1e2a36'
C_TXT, C_DIM = '#dbe4ee', '#7b8794'
C_CYAN, C_AMB, C_MAG = '#22d3ee', '#fbbf24', '#f472b6'
C_GRN, C_RED, C_RWY = '#4ade80', '#f87171', '#39424f'

plt.rcParams.update({
    'figure.facecolor': C_BG, 'axes.facecolor': C_PANEL,
    'savefig.facecolor': C_BG, 'text.color': C_TXT,
    'axes.labelcolor': C_TXT, 'xtick.color': C_DIM, 'ytick.color': C_DIM,
    'axes.edgecolor': C_GRID, 'grid.color': C_GRID, 'grid.alpha': 0.55,
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.titleweight': 'bold',
    'legend.framealpha': 0.0, 'axes.grid': True, 'grid.linewidth': 0.6,
})

def phase_color(phase: str) -> str:
    return {"CRAB": C_CYAN, "DECRAB/WING_LOW": C_AMB, "FLARE": C_GRN, "TOUCHDOWN": C_MAG}.get(phase, "#555555")

def aircraft_polygon(north: float, east: float, yaw: float, size: float = 18.0) -> np.ndarray:
    forward = np.array([math.sin(yaw), math.cos(yaw)])
    right = np.array([math.cos(yaw), -math.sin(yaw)])
    center = np.array([east, north])
    return np.vstack([center + size * forward, center - 0.65 * size * forward + 0.48 * size * right, center - 0.65 * size * forward - 0.48 * size * right])

def configure_plan_axes(ax: plt.Axes, result: SimulationResult) -> None:
    n, e = result.derived["north"], result.derived["east"]
    n_min, n_max = min(-80.0, float(np.min(n)) - 50.0), max(500.0, float(np.max(n)) + 50.0)
    e_lim = max(80.0, float(np.max(np.abs(e))) + 50.0)
    ax.set_xlim(-e_lim, e_lim)
    ax.set_ylim(n_min, n_max)
    ax.set_xlabel("East E [m]")
    ax.set_ylabel("North N [m]")
    ax.grid(True, alpha=0.25)

def draw_plan_background(ax: plt.Axes, result: SimulationResult) -> None:
    n = result.derived["north"]
    e_lim = max(80.0, float(np.max(np.abs(result.derived["east"]))) + 50.0)
    ax.add_patch(Rectangle((-12.0, -20.0), 24.0, 850.0, color=C_RWY, alpha=0.35, zorder=0))
    ax.plot([0.0, 0.0], [-20.0, 830.0], "w--", lw=0.9, alpha=0.8)
    w = result.wind_east
    if abs(w) > 1.0e-6:
        arrow_len = 12.0 * np.sign(w)
        for north in np.linspace(max(-20.0, float(np.min(n))), min(800.0, float(np.max(n))), 5):
            ax.arrow(-e_lim * 0.82, north, arrow_len, 0.0, width=0.01, head_width=4.0, head_length=4.0, color=C_RED, alpha=0.75, length_includes_head=True)
    ax.axvline(0.0, color="k", lw=0.5, alpha=0.5)

def plot_3d_approach(result: SimulationResult, output: Path) -> None:
    d = result.derived
    fig = plt.figure(figsize=(10, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(d["east"], d["north"], d["altitude"], color=C_CYAN, lw=2.0, label="aircraft path")
    runway_n = np.linspace(-20.0, 850.0, 2)
    ax.plot(np.zeros_like(runway_n), runway_n, np.zeros_like(runway_n), color=C_RWY, lw=12, alpha=0.65)
    ax.plot(np.zeros_like(runway_n), runway_n, np.zeros_like(runway_n) + 0.02, "w--", lw=1.0)
    if abs(result.wind_east) > 1.0e-6:
        ax.quiver(-40.0, 350.0, 2.0, np.sign(result.wind_east) * 22.0, 0.0, 0.0, color=C_RED, arrow_length_ratio=0.15, label="wind velocity")
    ax.set_xlabel("East E [m]")
    ax.set_ylabel("North N [m]")
    ax.set_zlabel("Altitude -D [m]")
    ax.set_title(f"3D runway and approach | wind velocity E = {result.wind_east:g} m/s")
    ax.view_init(elev=23.0, azim=-63.0)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_time_history(result: SimulationResult, output: Path) -> None:
    t, d, u = result.time, result.derived, result.controls
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    axes[0, 0].plot(t, d["altitude"], color=C_CYAN)
    axes[0, 0].set_ylabel("Altitude [m]")
    axes[0, 0].set_title("Position / speed")
    axes[0, 1].plot(t, d["air_speed"], label="air", color=C_GRN)
    axes[0, 1].plot(t, d["ground_speed"], label="ground", color=C_AMB)
    axes[0, 1].set_ylabel("Speed [m/s]")
    axes[0, 1].legend(frameon=False)
    axes[1, 0].plot(t, d["phi"] / DEG, label="roll phi")
    axes[1, 0].plot(t, d["theta"] / DEG, label="pitch theta")
    axes[1, 0].plot(t, d["psi"] / DEG, label="yaw psi")
    axes[1, 0].set_ylabel("Euler angle [deg]")
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 1].plot(t, d["alpha"] / DEG, label="alpha")
    axes[1, 1].plot(t, d["beta"] / DEG, label="beta")
    axes[1, 1].set_ylabel("Air angles [deg]")
    axes[1, 1].legend(frameon=False, fontsize=8)
    names = ["aileron", "elevator", "rudder", "throttle"]
    styles = [C_CYAN, C_MAG, C_AMB, C_GRN]
    for j in range(4):
        axes[2, 0].plot(t, u[:, j] / DEG if j < 3 else u[:, j], label=names[j], color=styles[j])
    axes[2, 0].set_ylabel("Controls [deg / fraction]")
    axes[2, 0].set_xlabel("Time [s]")
    axes[2, 0].legend(frameon=False, fontsize=8, ncol=2)
    axes[2, 1].plot(t, d["p"] / DEG, label="p")
    axes[2, 1].plot(t, d["q"] / DEG, label="q")
    axes[2, 1].plot(t, d["r"] / DEG, label="r")
    axes[2, 1].set_ylabel("Body rate [deg/s]")
    axes[2, 1].set_xlabel("Time [s]")
    axes[2, 1].legend(frameon=False, fontsize=8)
    for ax in axes.flat: ax.grid(True, alpha=0.25)
    fig.suptitle(f"6-DOF time history | E wind {result.wind_east:g} m/s | {result.metrics['outcome']}", y=0.995)
    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

def plot_animation(result: SimulationResult, output: Path, max_frames: int = 80) -> Optional[Path]:
    total = len(result.time)
    frame_count = min(max_frames, max(1, total))
    frame_indices = np.unique(np.linspace(0, total - 1, frame_count).astype(int))
    fig = plt.figure(figsize=(12, 6.5))
    grid = fig.add_gridspec(2, 2, width_ratios=[1.45, 1.0], height_ratios=[1.0, 1.0])
    ax_plan = fig.add_subplot(grid[:, 0])
    ax_side = fig.add_subplot(grid[0, 1])
    ax_hud = fig.add_subplot(grid[1, 1])
    configure_plan_axes(ax_plan, result)
    e, n = result.derived["east"], result.derived["north"]
    ax_side.set_xlim(float(np.min(n)) - 40.0, max(500.0, float(np.max(n)) + 40.0))
    ax_side.set_ylim(-2.0, max(80.0, float(np.max(result.derived["altitude"])) + 10.0))
    ax_side.set_xlabel("North N [m]")
    ax_side.set_ylabel("Altitude [m]")
    ax_side.grid(True, alpha=0.25)
    ax_hud.axis("off")
    def update(frame_number: int) -> Iterable[object]:
        idx = int(frame_indices[frame_number])
        ax_plan.clear(); ax_side.clear(); ax_hud.clear(); ax_hud.axis("off")
        configure_plan_axes(ax_plan, result)
        draw_plan_background(ax_plan, result)
        path_start = 0
        while path_start <= idx:
            path_phase = result.phases[path_start]
            path_end = path_start
            while path_end < idx and result.phases[path_end + 1] == path_phase: path_end += 1
            age_alpha = 0.28 + 0.72 * path_end / max(1, idx)
            ax_plan.plot(e[path_start : path_end + 1], n[path_start : path_end + 1], color=phase_color(path_phase), lw=2.2, alpha=age_alpha)
            path_start = path_end + 1
        yaw, bank = result.derived["psi"][idx], result.derived["phi"][idx]
        marker = Polygon(aircraft_polygon(n[idx], e[idx], yaw), closed=True, facecolor=phase_color(result.phases[idx]), edgecolor="white", zorder=5)
        ax_plan.add_patch(marker)
        ax_plan.plot([e[idx], e[idx] + 22.0 * math.sin(yaw)], [n[idx], n[idx] + 22.0 * math.cos(yaw)], color="white", lw=1.5)
        wing_half = 10.0 * max(0.25, math.cos(bank))
        wing_right = np.array([math.cos(yaw), -math.sin(yaw)])
        ax_plan.plot([e[idx] - wing_half * wing_right[0], e[idx] + wing_half * wing_right[0]], [n[idx] - wing_half * wing_right[1], n[idx] + wing_half * wing_right[1]], color="white", lw=2.0)
        ax_plan.set_title("Plan view: runway, path, heading")
        ax_side.plot(n[: idx + 1], result.derived["altitude"][: idx + 1], color=C_CYAN, lw=2.2)
        ax_side.axhline(0.0, color=C_RWY, lw=8, alpha=0.55)
        ax_side.scatter([n[idx]], [result.derived["altitude"][idx]], s=50, color=phase_color(result.phases[idx]), edgecolor="white", zorder=4)
        ax_side.set_xlim(float(np.min(n)) - 40.0, max(500.0, float(np.max(n)) + 40.0))
        ax_side.set_ylim(-2.0, max(80.0, float(np.max(result.derived["altitude"])) + 10.0))
        ax_side.set_xlabel("North N [m]"); ax_side.set_ylabel("Altitude [m]"); ax_side.grid(True, alpha=0.25)
        ax_side.set_title("Side view: 3 degree approach and flare")
        ax_hud.text(0.03, 0.92, "CROSSWIND LANDING\n\n" f"phase: {result.phases[idx]}\n" f"time: {result.time[idx]:6.2f} s\n" f"wind E: {result.wind_east:6.2f} m/s\n" f"height: {result.derived['altitude'][idx]:6.1f} m\n" f"lateral E: {result.derived['east'][idx]:6.1f} m\n" f"air speed: {result.derived['air_speed'][idx]:5.1f} m/s\n" f"alpha/beta: {result.derived['alpha'][idx] / DEG:5.1f} / " f"{result.derived['beta'][idx] / DEG:5.1f} deg\n" f"heading/bank: {result.derived['psi'][idx] / DEG:5.1f} / " f"{result.derived['phi'][idx] / DEG:5.1f} deg", va="top", fontsize=11, family="monospace", color=phase_color(result.phases[idx]))
        return []
    animation = FuncAnimation(fig, update, frames=len(frame_indices), interval=120, blit=False, repeat=True)
    frame_png = output.with_name(output.stem + "_frame.png")
    update(0)
    fig.savefig(frame_png, dpi=130)
    try:
        animation.save(output, writer=PillowWriter(fps=8))
        saved = output
    except Exception as exc:
        print(f"GIF not written ({exc}); retained {frame_png}", file=sys.stderr)
        saved = frame_png
    plt.close(fig)
    return saved

def write_summary_csv(results: Sequence[SimulationResult], trim: Dict[str, float], output: Path) -> None:
    fields = ["wind_east_mps", "outcome", "failure_reason", "criteria_pass", "criteria_failures", "touchdown_time_s", "touchdown_n_m", "touchdown_e_m", "touchdown_speed_mps", "touchdown_sink_rate_mps", "touchdown_heading_error_deg", "touchdown_bank_deg", "max_bank_deg", "max_sideslip_deg", "samples", "trim_residual_ok", "phase_checks_pass", "phase_decrab_alt_m", "phase_flare_alt_m", "wind_triangle_crab_deg", "wind_triangle_error_mps", "saturation_aileron_count", "saturation_elevator_count", "saturation_rudder_count", "saturation_throttle_count", "final_saturation_count", "trim_alpha_deg", "trim_elevator_deg", "trim_throttle", "trim_residual_norm"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            m = result.metrics
            row = {"wind_east_mps": result.wind_east, **{field: m.get(field, "") for field in fields if field in m}, "trim_alpha_deg": trim["alpha"] / DEG, "trim_elevator_deg": trim["elevator"] / DEG, "trim_throttle": trim["throttle"], "trim_residual_norm": trim["residual_norm"]}
            writer.writerow(row)

def plot_sweep(results: Sequence[SimulationResult], output: Path) -> None:
    winds = np.array([r.wind_east for r in results], dtype=float)
    metrics = [r.metrics for r in results]
    e = np.array([m["touchdown_e_m"] for m in metrics], dtype=float)
    sink = np.array([m["touchdown_sink_rate_mps"] for m in metrics], dtype=float)
    bank = np.array([m["touchdown_bank_deg"] for m in metrics], dtype=float)
    success = np.array([bool(m["criteria_pass"]) for m in metrics])
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    axes[0].plot(winds, e, "o-", color=C_CYAN, label="touchdown E (only landed cases)")
    axes[0].axhspan(-MAX_TOUCHDOWN_OFFSET_M, MAX_TOUCHDOWN_OFFSET_M, color=C_GRN, alpha=0.12, label=f"criteria +/-{MAX_TOUCHDOWN_OFFSET_M:g} m")
    axes[0].axhline(0.0, color="k", lw=0.7)
    axes[0].set_ylabel("Touchdown lateral E [m]")
    axes[0].set_title("Crosswind sweep and model constraints")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(winds, np.abs(sink), "o-", color=C_RED, label="|touchdown sink rate|")
    axes[1].plot(winds, bank, "s-", color=C_MAG, label="|touchdown bank|")
    axes[1].axhline(MAX_TOUCHDOWN_SINK_MPS, color=C_RED, ls="--", alpha=0.7, label=f"sink limit {MAX_TOUCHDOWN_SINK_MPS:g} m/s")
    axes[1].axhline(MAX_TOUCHDOWN_BANK_DEG, color=C_MAG, ls="--", alpha=0.7, label=f"bank limit {MAX_TOUCHDOWN_BANK_DEG:g} deg")
    axes[1].set_xlabel("Steady wind velocity in +E [m/s]")
    axes[1].set_ylabel("Magnitude [m/s or deg]")
    axes[1].legend(frameon=False, fontsize=8)
    for i, ok in enumerate(success):
        if not ok: axes[0].annotate("FAIL", (winds[i], 0.0), xytext=(0, 10), textcoords="offset points", ha="center", color=C_RED, fontsize=8)
    for ax in axes: ax.grid(True, alpha=0.25)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(output, dpi=150)
    plt.close(fig)

def print_mission_report(results: Sequence[SimulationResult], trim: Dict[str, float], output_dir: Path):
    print("\n" + "="*60)
    print("                MISSION REPORT SUMMARY")
    print("="*60)
    print(f"Aircraft Model: Generic Light Aircraft (1000kg)")
    print(f"Trim Solver: {trim['solver']} (Residual: {trim['residual_norm']:.2e})")
    print("-"*60)
    for r in results:
        m = r.metrics
        status = "[PASS]" if m['criteria_pass'] else "[FAIL]"
        print(f"\nWIND: {r.wind_east:.1f} m/s ({r.wind_east*1.944:.1f} kt) {status}")
        print(f"  Outcome: {m['outcome']}")
        if m['touchdown']:
            print(f"  Touchdown: N={m['touchdown_n_m']:.1f}m, E={m['touchdown_e_m']:.1f}m")
            print(f"  Sink Rate: {m['touchdown_sink_rate_mps']:.2f} m/s  |  Bank: {m['touchdown_bank_deg']:.1f} deg")
        if m['failure_reason']:
            print(f"  Reason: {m['failure_reason']}")
    print("-"*60)
    print(f"Output Directory: {output_dir.resolve()}")
    print("="*60 + "\n")