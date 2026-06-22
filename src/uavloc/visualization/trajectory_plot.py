from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

def plot_xy_trajectory(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8,6))
    plt.plot(df["x_enu_m"], df["y_enu_m"], linewidth = 1.2)
    plt.scatter(df["x_enu_m"].iloc[0], df["y_enu_m"].iloc[0], marker="o", s=60, label="start")
    plt.scatter(df["x_enu_m"].iloc[-1], df["y_enu_m"].iloc[-1], marker="x", s=60, label="End")
    plt.xlabel("X ENU [m]")
    plt.ylabel("Y ENU [m]")
    plt.title("Reference Trajectory (XY)")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_altitude_profile(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index

    plt.figure(figsize=(9,4))
    plt.plot(x_axis, df["alt"], linewidth=1.2)
    plt.xlabel("Timestamp")
    plt.ylabel("Altitude [m]")
    plt.title("Reference Altitude Profile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

def plot_speed_profile(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    speed_cols = ["vel_n_m_s", "vel_e_m_s", "vel_d_m_s"]
    missing = [c for c in speed_cols if c not in df.columns]

    if missing:
        raise ValueError(f"Missing speed columns: {missing}")

    speed = (df["vel_n_m_s"] ** 2 + df["vel_e_m_s"] ** 2 + df["vel_d_m_s"] ** 2) ** 0.5
    x_axis = df["timestamp"] if "timestamp" in df.columns else df.index

    plt.figure(figsize=(9, 4))
    plt.plot(x_axis, speed, linewidth=1.2)
    plt.xlabel("Timestamp")
    plt.ylabel("Speed [m/s]")
    plt.title("Reference Speed Profile")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

