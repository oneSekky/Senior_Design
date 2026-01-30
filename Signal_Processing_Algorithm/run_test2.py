"""
Run trajectory reconstruction on Test2.csv (Hello World)
"""

import matplotlib.pyplot as plt
from data_reader import read_accelerometer_csv
from stroke_segmentation import segment_strokes
from trajectory_reconstruction import (
    export_trajectory_json,
    plot_trajectory,
    reconstruct_stroke_trajectory,
    reconstruct_trajectory,
    save_trajectory_svg,
)


def main():
    print("=" * 70)
    print("Trajectory Reconstruction - Test2.csv (Hello World)")
    print("=" * 70)

    # Load data
    print("\n1. Loading data...")
    df = read_accelerometer_csv("../Test_Data/Test2.csv")
    print(f"   Loaded {len(df)} samples")
    print(
        f"   Duration: {(df['time[us]'].iloc[-1] - df['time[us]'].iloc[0]) / 1e6:.2f} seconds"
    )

    # Reconstruct full trajectory
    print("\n2. Reconstructing full trajectory...")
    trajectory, velocity, time = reconstruct_trajectory(df, cutoff_freq=3.0)
    print(f"   Trajectory points: {len(trajectory)}")
    print(f"   X range: {trajectory[:, 0].min():.4f} to {trajectory[:, 0].max():.4f} m")
    print(f"   Y range: {trajectory[:, 1].min():.4f} to {trajectory[:, 1].max():.4f} m")

    # Save full trajectory
    print("\n3. Saving full trajectory...")
    save_trajectory_svg(trajectory, "test2_full_trajectory.svg")
    export_trajectory_json(trajectory, "test2_full_trajectory.json")

    # Plot full trajectory
    plt_full = plot_trajectory(trajectory, "Test2 - Full Trajectory (Hello World)")
    plt_full.savefig("test2_full_trajectory.png", dpi=150, bbox_inches="tight")
    print("   Saved test2_full_trajectory.png")
    plt_full.close()

    # Segment into individual strokes
    print("\n4. Segmenting into individual strokes...")
    strokes = segment_strokes(df, threshold=200, min_samples=20, merge_gap=100)
    print(f"   Found {len(strokes)} strokes")

    # Reconstruct each stroke
    print("\n5. Reconstructing individual strokes...")
    trajectories = []
    for i, stroke in enumerate(strokes):
        traj = reconstruct_stroke_trajectory(stroke["data"], cutoff_freq=3.0)
        trajectories.append(traj)

        # Save each stroke as SVG
        save_trajectory_svg(traj, f"test2_stroke_{i + 1:02d}.svg")

        print(
            f"   Stroke {i + 1:2d}: {len(traj):4d} points, "
            f"duration {stroke['duration']:.2f}s, "
            f"X:[{traj[:, 0].min():7.4f}, {traj[:, 0].max():7.4f}], "
            f"Y:[{traj[:, 1].min():7.4f}, {traj[:, 1].max():7.4f}] m"
        )

    # Plot all strokes
    print("\n6. Creating visualization of all strokes...")
    n_strokes = len(trajectories)
    cols = 5
    rows = (n_strokes + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(20, 4 * rows))
    axes = axes.flatten() if n_strokes > 1 else [axes]

    for i, traj in enumerate(trajectories):
        axes[i].plot(traj[:, 0], traj[:, 1], "b-", linewidth=2)
        axes[i].plot(traj[0, 0], traj[0, 1], "go", markersize=8, label="Start")
        axes[i].plot(traj[-1, 0], traj[-1, 1], "ro", markersize=8, label="End")
        axes[i].set_title(f"Stroke {i + 1}", fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].axis("equal")
        axes[i].legend(fontsize=8)

    # Hide unused subplots
    for i in range(n_strokes, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    plt.savefig("test2_all_strokes.png", dpi=150, bbox_inches="tight")
    print("   Saved test2_all_strokes.png")
    plt.close()

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print("\nGenerated files:")
    print("  • test2_full_trajectory.png/svg/json - Complete writing session")
    print("  • test2_all_strokes.png - Grid view of all individual strokes")
    print(
        f"  • test2_stroke_01.svg through test2_stroke_{len(strokes):02d}.svg - Individual strokes"
    )
    print("\nThe SVG files can be opened in:")
    print("  - Adobe Illustrator")
    print("  - Inkscape")
    print("  - Any web browser")
    print("  - Web design tools")
    print("\nThe JSON file contains raw coordinate data for programming.")
    print("=" * 70)


if __name__ == "__main__":
    main()
