from pathlib import Path
from typing import Optional, Sequence, List

from src.aircraft import AircraftModel
from src.dynamics import solve_trim
from src.simulation import simulate_landing, SimulationResult
from src.visualization import (
    plot_3d_approach,
    plot_time_history,
    plot_animation,
    write_summary_csv,
    plot_sweep,
    print_mission_report,
)

def get_user_inputs():
    print("\n--- CROSSWIND LANDING SIMULATION SETUP ---")
    try:
        wind_in = input("Enter Crosswind Speed (m/s) [Default 6.0]: ").strip()
        wind = float(wind_in) if wind_in else 6.0
        sweep_in = input("Enter Wind Sweep (comma-separated, e.g. 0,5,10) [Default 0,5,10]: ").strip()
        sweep = [float(x) for x in sweep_in.split(',')] if sweep_in else [0.0, 5.0, 10.0]
        duration_in = input("Enter Duration (s) [Default 55.0]: ").strip()
        duration = float(duration_in) if duration_in else 55.0
        quick_in = input("Quick Preview Mode? (y/n) [Default n]: ").strip().lower()
        quick = quick_in == 'y'
        print("--------------------------------------\n")
        return wind, sweep, duration, quick
    except ValueError:
        print("Invalid input. Using defaults.")
        return 6.0, [0.0, 5.0, 10.0], 55.0, False

def main(argv: Optional[Sequence[str]] = None) -> int:
    wind_arg, sweep_arg, duration_arg, quick_arg = get_user_inputs()
    duration = 50.0 if quick_arg else duration_arg
    dt = 0.05 if quick_arg else 0.02
    output_dir = Path("aircraft_crosswind_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    model = AircraftModel()
    trim = solve_trim(model)
    print("6-DOF crosswind landing exhibit (Refined)")
    print(f"Trim solver: {trim['solver']}; residual norm = {trim['residual_norm']:.6g}")
    print(f"Simulating animated case at +E wind {wind_arg:g} m/s ...")
    animated_result = simulate_landing(wind_arg, duration, dt, model, trim)
    plot_3d_approach(animated_result, output_dir / "runway_approach_3d.png")
    plot_time_history(animated_result, output_dir / "time_history.png")
    max_frames = 36 if quick_arg else 80
    plot_animation(animated_result, output_dir / "crosswind_landing.gif", max_frames=max_frames)
    print("Running wind sweep:")
    sweep_results: List[SimulationResult] = []
    for wind in sweep_arg:
        result = simulate_landing(wind, duration, dt, model, trim)
        sweep_results.append(result)
        m = result.metrics
        print(f"  +E {wind:g} m/s -> {m['outcome']}")
    write_summary_csv(sweep_results, trim, output_dir / "crosswind_summary.csv")
    plot_sweep(sweep_results, output_dir / "touchdown_sweep.png")
    print_mission_report(sweep_results, trim, output_dir)
    return 0

if __name__ == "__main__":
    raise SystemExit(main())