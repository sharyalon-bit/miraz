"""
Simulation of a laser beam refracting through an aquarium that contains a
vertical gradient of sugar concentration.

The aquarium is divided into ``num_layers`` horizontal layers. The refractive
index varies linearly with depth: the bottom layer has the highest index
(highest sugar concentration) and the top layer has the lowest index. The
laser enters from the air above the water surface at a user-defined angle
(measured from the vertical normal) and refracts at every layer boundary
according to Snell's law.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


N_AIR = 1.0  # Refractive index of air


def simulate_laser_refraction(
    water_height=20.0,
    num_layers=40,
    n_top=1.333,
    n_bottom=1.49,
    entry_angle_deg=60.0,
    aquarium_width=30.0,
    show_plot=True,
):
    """
    Trace a laser beam through a sugar-gradient aquarium.

    Parameters
    ----------
    water_height : float
        Height of the water column in the aquarium [cm].
    num_layers : int
        Number of horizontal layers (>=1).
    n_top : float
        Refractive index of the uppermost layer (lowest sugar concentration).
    n_bottom : float
        Refractive index of the lowest layer (highest sugar concentration).
    entry_angle_deg : float
        Angle of the incident laser measured from the vertical normal, in
        degrees. Positive values bend the beam to the right.
    aquarium_width : float
        Horizontal width of the aquarium [cm], used for the drawing.
    show_plot : bool
        If True, display the matplotlib figure.

    Returns
    -------
    xs, ys : np.ndarray
        Points along the laser path.
    n_layers : np.ndarray
        Refractive indices of each layer (top -> bottom).
    thetas_deg : np.ndarray
        Angle of the beam in each layer [deg from normal].
    """
    if num_layers < 1:
        raise ValueError("num_layers must be at least 1")
    if not (0 <= entry_angle_deg < 90):
        raise ValueError("entry_angle_deg must be in [0, 90) degrees")
    if n_top <= 0 or n_bottom <= 0:
        raise ValueError("refractive indices must be positive")

    # Linearly interpolated refractive index, top -> bottom
    if num_layers == 1:
        n_layers = np.array([(n_top + n_bottom) / 2.0])
    else:
        n_layers = np.linspace(n_top, n_bottom, num_layers)

    layer_thickness = water_height / num_layers
    entry_angle = np.radians(entry_angle_deg)

    # Snell's law: n_air * sin(theta_air) = n_i * sin(theta_i) for every layer
    sin_thetas = N_AIR * np.sin(entry_angle) / n_layers
    sin_thetas = np.clip(sin_thetas, -1.0, 1.0)
    thetas = np.arcsin(sin_thetas)

    # -------------------------------------------------------------------
    # Trace the beam
    # -------------------------------------------------------------------
    # Show a small segment in the air above the surface for visualization.
    air_length = max(0.2 * water_height, 3.0)
    # Entry point on water surface (arbitrary x so the beam fits in the tank)
    x_entry = 0.25 * aquarium_width
    y_entry = water_height

    x_start = x_entry - air_length * np.sin(entry_angle)
    y_start = y_entry + air_length * np.cos(entry_angle)

    xs = [x_start, x_entry]
    ys = [y_start, y_entry]

    x, y = x_entry, y_entry
    exited_side = False
    for theta in thetas:
        dx = layer_thickness * np.tan(theta)
        dy = -layer_thickness
        new_x = x + dx
        new_y = y + dy

        # Stop if the beam leaves the aquarium through a side wall
        if new_x > aquarium_width:
            frac = (aquarium_width - x) / dx if dx != 0 else 1.0
            x = aquarium_width
            y = y + frac * dy
            xs.append(x)
            ys.append(y)
            exited_side = True
            break
        if new_x < 0:
            frac = (0 - x) / dx if dx != 0 else 1.0
            x = 0.0
            y = y + frac * dy
            xs.append(x)
            ys.append(y)
            exited_side = True
            break

        x, y = new_x, new_y
        xs.append(x)
        ys.append(y)

    xs = np.array(xs)
    ys = np.array(ys)

    # -------------------------------------------------------------------
    # Plot
    # -------------------------------------------------------------------
    if show_plot:
        _plot_simulation(
            xs, ys, n_layers, layer_thickness,
            water_height, aquarium_width, entry_angle_deg,
            n_top, n_bottom, x_entry, exited_side, thetas,
        )

    return xs, ys, n_layers, np.degrees(thetas)


def _plot_simulation(xs, ys, n_layers, layer_thickness,
                     water_height, aquarium_width, entry_angle_deg,
                     n_top, n_bottom, x_entry, exited_side, thetas):
    num_layers = len(n_layers)
    fig, ax = plt.subplots(figsize=(11, 8))

    # Color each layer according to its refractive index
    cmap = plt.cm.BuPu
    n_min, n_max = min(n_top, n_bottom), max(n_top, n_bottom)
    n_range = n_max - n_min if n_max != n_min else 1.0

    for i, n in enumerate(n_layers):
        y_top = water_height - i * layer_thickness
        y_bot = y_top - layer_thickness
        shade = 0.15 + 0.55 * (n - n_min) / n_range
        ax.add_patch(Rectangle(
            (0, y_bot), aquarium_width, layer_thickness,
            facecolor=cmap(shade), edgecolor='lightgray',
            linewidth=0.4, zorder=1,
        ))
        ax.text(aquarium_width + 0.4, (y_top + y_bot) / 2.0,
                f"n={n:.4f}", va='center', fontsize=8, color='black')

    # Aquarium walls (glass)
    ax.plot([0, 0], [0, water_height + 0.2], 'k-', linewidth=2.5, zorder=2)
    ax.plot([aquarium_width, aquarium_width], [0, water_height + 0.2],
            'k-', linewidth=2.5, zorder=2)
    ax.plot([0, aquarium_width], [0, 0], 'k-', linewidth=2.5, zorder=2)

    # Water surface
    ax.plot([0, aquarium_width], [water_height, water_height],
            color='steelblue', linewidth=1.5, zorder=2)

    # Laser path
    ax.plot(xs, ys, color='red', linewidth=2.2, label='Laser beam', zorder=5)
    # Mark every refraction point (interface between layers)
    ax.scatter(xs[1:-1], ys[1:-1], color='yellow', edgecolor='darkred',
               s=22, zorder=6, label='Refraction point')
    ax.plot(xs[0], ys[0], 'o', color='red', markersize=7, zorder=6)
    ax.annotate('Laser source', (xs[0], ys[0]),
                xytext=(8, 8), textcoords='offset points', fontsize=9)

    # Annotate the angle inside the first, middle and last layer
    annot_idx = sorted({0, len(thetas) // 2, len(thetas) - 1})
    for i in annot_idx:
        seg_x = 0.5 * (xs[i + 1] + xs[i + 2])
        seg_y = 0.5 * (ys[i + 1] + ys[i + 2])
        ax.annotate(f"θ={np.degrees(thetas[i]):.1f}°",
                    (seg_x, seg_y),
                    xytext=(10, 0), textcoords='offset points',
                    fontsize=8, color='darkred',
                    arrowprops=dict(arrowstyle='-', color='darkred',
                                    lw=0.6, alpha=0.6))

    # Normal line at entry
    normal_len = 0.15 * water_height
    ax.plot([x_entry, x_entry],
            [water_height - normal_len, water_height + normal_len],
            'k--', linewidth=0.8, alpha=0.6, zorder=3)
    ax.text(x_entry + 0.2, water_height + normal_len, 'normal',
            fontsize=8, alpha=0.7)

    exit_str = "side wall" if exited_side else "bottom"
    ax.set_title(
        f"Laser refraction through sugar-gradient aquarium\n"
        f"entry angle = {entry_angle_deg}°   |   layers = {num_layers}   |   "
        f"n: {n_top} (top) → {n_bottom} (bottom)   |   exits through {exit_str}"
    )
    ax.set_xlabel('x [cm]')
    ax.set_ylabel('y [cm]')

    margin = max(2.0, 0.1 * aquarium_width)
    ax.set_xlim(-margin, aquarium_width + 6)
    ax.set_ylim(-margin, water_height + 0.35 * water_height)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.25)
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()


def _prompt(message, default, cast):
    raw = input(f"{message} [{default}]: ").strip()
    if raw == "":
        return default
    try:
        return cast(raw)
    except ValueError:
        print(f"  Invalid input, using default {default}.")
        return default


def get_user_input():
    """Read simulation parameters from stdin, with defaults."""
    print("=" * 60)
    print("  Laser refraction in a sugar-gradient aquarium")
    print("=" * 60)
    print("Press ENTER to accept the default value shown in brackets.\n")

    water_height = _prompt("Water height [cm]", 20.0, float)
    num_layers = _prompt("Number of layers", 10, int)
    n_top = _prompt("Refractive index at top (low sugar)", 1.333, float)
    n_bottom = _prompt("Refractive index at bottom (high sugar)", 1.49, float)
    entry_angle_deg = _prompt("Laser entry angle from vertical [deg]",
                              30.0, float)

    return dict(
        water_height=water_height,
        num_layers=num_layers,
        n_top=n_top,
        n_bottom=n_bottom,
        entry_angle_deg=entry_angle_deg,
    )


if __name__ == "__main__":
    params = get_user_input()
    xs, ys, n_layers, thetas_deg = simulate_laser_refraction(**params)

    print("\n--- Results ---")
    for i, (n, t) in enumerate(zip(n_layers, thetas_deg)):
        print(f"  Layer {i + 1:2d}: n = {n:.4f}   angle from normal = {t:6.2f}°")
    print(f"\n  Final beam position: x = {xs[-1]:.2f} cm, y = {ys[-1]:.2f} cm")
