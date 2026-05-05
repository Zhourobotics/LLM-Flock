import os
import shutil
import matplotlib

matplotlib.use("Agg")

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import colorsys


# Use a white background style with grid lines.
# Style names differ across Matplotlib versions, so pick the best available match.
for candidate_style in ("seaborn-v0_8-whitegrid", "seaborn-whitegrid", "ggplot"):
    if candidate_style in plt.style.available:
        plt.style.use(candidate_style)
        break
else:
    plt.style.use("default")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["figure.facecolor"] = "white"

# LaTeX text rendering is opt-in because partial TeX installs often break plotting.
# Enable only when FLOCKING_USE_TEX=1 and a latex binary is available.
LATEX_REQUESTED = os.environ.get("FLOCKING_USE_TEX", "0").lower() in {
    "1",
    "true",
    "yes",
}
LATEX_AVAILABLE = shutil.which("latex") is not None
USE_TEX = LATEX_REQUESTED and LATEX_AVAILABLE
if LATEX_REQUESTED and not LATEX_AVAILABLE:
    print(
        "FLOCKING_USE_TEX requested but 'latex' was not found. "
        "Falling back to Matplotlib text rendering."
    )
plt.rcParams.update(
    {
        "text.usetex": USE_TEX,
        "font.family": "serif",  # (optional) LaTeX-style font
        # "font.serif": ["Computer Modern"],  # (optional) Computer Modern font
        # "font.serif": [
        #     "Times",
        #     "Palatino",
        #     "New Century Schoolbook",
        #     "Bookman",
        #     "Computer Modern Roman",
        # ],  # Fallback fonts
        "axes.titlesize": 22,  # Title font size
        "axes.labelsize": 18,  # X and Y label font size
        "xtick.labelsize": 16,  # X axis tick font size
        "ytick.labelsize": 16,  # Y axis tick font size
        "legend.fontsize": 14,  # Legend font size
        "font.size": 18,  # General default font size
    }
)


class Graph:
    """
    A class to facilitate drawing graphs of drone movements in a 2D plane
    """

    colors = (
        np.array(
            [
                [0xE8, 0x3B, 0x3B],  # red
                [0x7A, 0x30, 0x45],  # maroon
                [0xF9, 0xC2, 0x2B],  # yellow
                [0x16, 0x5A, 0x4C],  # forest green
                [0x4D, 0x9B, 0xE6],  # blue
                [0x30, 0xE1, 0xB9],  # cyan
                [0xE3, 0x77, 0xC2],  # pink
                [0xFF, 0x7F, 0x0E],  # orange
                [0x94, 0x67, 0xBD],  # purple
                [0x7F, 0x7F, 0x7F],  # gray
            ]
        )
        / 255
    )

    @staticmethod
    def generate_distinct_colors(n):
        """Generate n visually distinct colors for plan source visualization"""
        # Use strongly distinct colors for better visual differentiation
        colors = []
        for i in range(n):
            # Use HSV color space with higher saturation for more distinct colors
            hue = i / n
            saturation = 0.9  # Increase from 0.7 to 0.9 for more vivid colors
            value = 0.9
            colors.append(colorsys.hsv_to_rgb(hue, saturation, value))
        return colors

    @staticmethod
    def plot_animated(data, show_influence=False):
        # Debug prints to verify data loading
        print("\n=== DATA DEBUG INFO ===")
        print(
            "First agent keys:",
            list(data.agents[0].keys()) if data.agents else "No agents",
        )

        # Check specifically for plan source/origin attributes and print them more prominently
        print("\n PLAN SOURCE DATA:")
        for i, agent in enumerate(data.agents):
            source_id = agent.get("plan_source_id", "MISSING")
            origin_id = agent.get("plan_origin_id", "MISSING")
            print(f"Agent {i}: source_id={source_id}, origin_id={origin_id}")

        # Use a square figure by setting figsize to (8, 8) for better visibility
        fig, ax = plt.subplots(figsize=(8, 8))
        ###### RADIUS ######
        radius = data.settings.safe_distance-2
        # radius = 6
        # Set both x and y limits
        ax.set_xlim(data.settings.x_min, data.settings.x_max)
        ax.set_ylim(data.settings.y_min, data.settings.y_max)
        # ax.set_xlabel("x")
        # ax.set_ylabel("y")
        ax.set_xticks(np.arange(data.settings.x_min, data.settings.x_max + 1, 20))
        ax.set_yticks(np.arange(data.settings.y_min, data.settings.y_max + 1, 20))
        # ax.set_title("Drone Trajectories")
        ax.grid(True, linestyle="--", alpha=0.6)

        # Ensure that the x and y axes are equally scaled
        ax.set_aspect("equal", adjustable="box")

        lines = []
        scatters = []
        safety_circles = []
        influence_texts = []  # New list to store influence text annotations
        # plan_source_text = None

        # Generate distinct colors with more contrast and print a clear reference
        num_agents = data.settings.agents
        plan_colors = Graph.generate_distinct_colors(num_agents)
        use_plan_origin_colors = data.settings.agent_mode in {"influence", "plan"}

        print("\n🎨 COLOR MAPPING:")
        for i, color in enumerate(plan_colors):
            r, g, b = [int(c * 255) for c in color]
            print(f"Agent {i}: RGB({r},{g},{b}) - hex: #{r:02x}{g:02x}{b:02x}")

        # Create lines, scatters, and safety circles for each agent
        colors_agent = {}
        for i in range(data.settings.agents):
            current_color = Graph.colors[i % len(Graph.colors)]
            colors_agent[i] = current_color

            # First create the lines for the trails (lowest z-order)
            (line,) = ax.plot(
                [],
                [],
                lw=2,
                color=current_color,
                linestyle="--",
                label=f"Drone {i + 1}",
                zorder=1,  # Set lowest z-order for trails
            )
            lines.append(line)

            # Next create circles (middle z-order)
            circle = plt.Circle(
                (0, 0),
                radius,
                color=colors_agent[i],
                alpha=0.3,
                fill=True,
                zorder=2,  # Set middle z-order for circles
            )
            ax.add_patch(circle)
            safety_circles.append(circle)

            # Finally create the scatter points (highest z-order)
            scatter = ax.scatter(
                [],
                [],
                marker="o",
                color=current_color,
                s=50,
                alpha=0.8,
                zorder=3,  # Set highest z-order for scatter points
            )
            scatters.append(scatter)

            # Add text annotation for influence if enabled
            if show_influence:
                infl_text = ax.text(
                    0,
                    0,
                    "",
                    fontsize=20,
                    ha="center",
                    va="bottom",
                    color="black",
                    fontweight="bold",
                    zorder=4,
                )
                influence_texts.append(infl_text)

        def calculate_influence(positions, agent_idx, frame):
            """Calculate influence score for agent based on its position relative to others"""
            # Get the current position of the agent
            curr_pos = np.array(positions[agent_idx][frame])
            influence_score = 0
            neighbor_count = 0
            # Calculate influence based on distance to other agents
            for i in range(len(positions)):
                if i == agent_idx:
                    continue

                other_pos = np.array(positions[i][frame])
                distance = np.linalg.norm(curr_pos - other_pos)

                # Simple inverse distance influence calculation
                if distance < data.settings.comm_range:  # Only count nearby agents
                    # influence_score += 1 / max(distance, 0.1)  # Avoid division by zero
                    neighbor_count += 1

            return round(neighbor_count / (data.settings.agents - 1), 2)

        def init():
            # Initialize empty data for each plot element
            for line, scatter in zip(lines, scatters):
                line.set_data([], [])
                scatter.set_offsets(np.empty((0, 2)))

            # Initialize safety circles at origin
            for circle in safety_circles:
                circle.center = (0, 0)

            # plan_source_text.set_text("")

            # ax.legend(
            #     loc="best",
            #     labelspacing=0.6,
            #     fontsize=18,
            #     edgecolor="black",
            #     facecolor="white",
            #     framealpha=1,
            #     frameon=True,
            # )
            # Initialize influence texts
            if show_influence:
                for text in influence_texts:
                    text.set_text("")
                    text.set_position((0, 0))

            return (
                lines
                + scatters
                + safety_circles
                + (influence_texts if show_influence else [])
            )

        # Calculate interpolated positions for smoother animation
        def get_interpolated_positions(positions, frame, interp_steps=10):
            """Generate smoothly interpolated positions between timesteps"""
            # Determine the actual simulation frame
            sim_frame = int(frame / interp_steps)
            # If we're at the end of the simulation data, just return the last position
            if sim_frame >= len(positions) - 1:
                return positions[-1]

            # Calculate the interpolation fraction
            frac = (frame % interp_steps) / interp_steps

            # Get the positions to interpolate between
            pos1 = np.array(positions[sim_frame])
            pos2 = np.array(positions[min(sim_frame + 1, len(positions) - 1)])

            # Linear interpolation
            return pos1 + frac * (pos2 - pos1)

        # Total frames will be simulation frames * interpolation steps
        interp_steps = 5  # Number of frames to interpolate between each simulation step
        total_frames = (data.settings.rounds - 1) * interp_steps + 1
        # total_frames = data.settings.rounds * interp_steps + 1

        def update(frame):
            # Track plan sources for visualization
            plan_sources = {}

            # Map animation frame to simulation frame
            sim_frame = int(frame / interp_steps)

            # Get all agent positions for this simulation frame for influence calculation
            if show_influence and sim_frame < data.settings.rounds:
            # if show_influence and sim_frame < data.settings.rounds+1:
                all_agent_positions = [
                    data.agents[i]["position_history"]
                    for i in range(data.settings.agents)
                ]
                # all_agent_positions[0].append([60, 60])
                # all_agent_positions[1].append(data.agents[1]["position_history"][-1])
                # all_agent_positions[2].append([60, 40])
                # all_agent_positions[3].append(data.agents[3]["position_history"][-1])

            for i, (line, scatter, circle) in enumerate(
                zip(lines, scatters, safety_circles)
            ):
                all_positions = data.agents[i]["position_history"]
                # if i == 0:
                #     all_positions.append([60, 60])
                # elif i == 1:
                #     all_positions.append(data.agents[i]["position_history"][-1])
                # elif i == 2:
                #     all_positions.append([60, 40])
                # elif i == 3:
                #     all_positions.append(data.agents[i]["position_history"][-1])
                # Get the interpolated position for this frame
                current_pos = get_interpolated_positions(
                    all_positions, frame, interp_steps
                )

                # Update the safety circle's center with interpolated position
                circle.center = current_pos

                # Update influence text if enabled
                # if show_influence and sim_frame < data.settings.rounds+1:
                if show_influence and sim_frame < data.settings.rounds:
                    # Initialize influence score
                    influence_score = 0

                    # Calculate influence only for actual simulation frames
                    if frame % interp_steps == 0:
                        influence_score = calculate_influence(
                            all_agent_positions, i, sim_frame
                        )
                    # For interpolated frames, use the last calculated value
                    elif sim_frame > 0:
                        # Use the value from the current simulation frame
                        influence_score = calculate_influence(
                            all_agent_positions, i, sim_frame
                        )

                    # Position text above the agent
                    influence_texts[i].set_position(
                        (
                            current_pos[0],
                            current_pos[1] + data.settings.safe_distance + 0.5,
                        )
                    )
                    influence_texts[i].set_text(f"{influence_score:.2f}")

                # Force update of circle color with direct matplotlib approach
                if sim_frame > 0:
                    origin_id = i
                    if use_plan_origin_colors:
                        plan_origin_history = data.agents[i].get("plan_origin", [])
                        if plan_origin_history:
                            # Get the origin ID for the corresponding simulation frame
                            origin_id = plan_origin_history[
                                min(sim_frame, len(plan_origin_history) - 1)
                            ]
                        if origin_id is None:
                            origin_id = i

                        # Ensure it's an integer and in range
                        try:
                            origin_id = int(origin_id)
                        except (ValueError, TypeError):
                            origin_id = i
                        if origin_id not in colors_agent:
                            origin_id = i

                    # Get the color for this origin
                    color = colors_agent[origin_id]

                    # Reset the circle completely to force color update
                    circle.remove()
                    circle = plt.Circle(
                        current_pos,
                        radius,
                        color=color,
                        alpha=0.5,
                        fill=True,
                        zorder=2,  # Maintain middle z-order for circles
                    )
                    ax.add_patch(circle)
                    safety_circles[i] = circle

                    # Track for display
                    if use_plan_origin_colors:
                        if origin_id in plan_sources:
                            plan_sources[origin_id].append(i)
                        else:
                            plan_sources[origin_id] = [i]

                # Create a smooth trail by using more positions
                trail_length = 15  # Longer trail for smoother effect

                # Calculate trail start position, handle edge cases
                start_frame = max(0, sim_frame - trail_length // interp_steps)

                # Get positions for the trail
                trail_positions = []
                for f in range(start_frame, sim_frame + 1):
                    if f < len(all_positions):
                        trail_positions.append(all_positions[f])

                # Add the current interpolated position
                if trail_positions and len(trail_positions) > 0:
                    trail_positions.append(current_pos)

                # Set the trail data
                if trail_positions:
                    xs = [pos[0] for pos in trail_positions]
                    ys = [pos[1] for pos in trail_positions]
                    line.set_data(xs, ys)

                # Update scatter with current interpolated position
                # Ensure scatter points remain above circles with higher zorder
                scatter.set_offsets([current_pos])
                scatter.set_zorder(3)  # Reinforce highest z-order for scatter points

            # Save key frames as images (only save actual simulation frames, not interpolated ones)
            if frame % interp_steps == 0:
                if sim_frame == data.settings.rounds - 1:
                # if sim_frame == data.settings.rounds:

                    plt.savefig(f"{data.directory}/last.svg", bbox_inches="tight")
                plt.savefig(
                    f"{data.directory}/frame_{sim_frame:02d}.pdf",
                    bbox_inches="tight",
                    dpi=300,
                )
                plt.savefig(
                    f"{data.directory}/frame_{sim_frame:02d}.png",
                    bbox_inches="tight",
                    dpi=300,
                )

            return (
                lines
                + scatters
                + safety_circles
                + (influence_texts if show_influence else [])
            )

        # Set tight layout before creating animation to minimize margins
        plt.tight_layout(pad=0.5)

        # Create a smoother animation with higher frame rate but slower playback
        ani = FuncAnimation(
            fig,
            update,
            frames=total_frames,
            init_func=init,
            blit=True,
            interval=200,  # Increased interval (ms) for slower animation
        )

        # Save animation with lower fps for slower playback and tight margins
        ani.save(
            f"{data.directory}/animation.gif",
            fps=8,  # Lower fps for slower animation
            dpi=300,
            writer="pillow",
            savefig_kwargs={"bbox_inches": "tight", "pad_inches": 0.1},
        )

        print("Animation saved successfully with tight margins!")
        plt.close()
        return ani
