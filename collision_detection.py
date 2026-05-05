"""
Collision Detection Module

This module provides comprehensive collision detection for multi-agent systems.
It checks two types of collisions:
1. Distance collision: Agents too close to each other
2. Trajectory collision: Agent paths coming within unsafe distance during movement
"""

import numpy as np
from typing import Dict, List, Tuple


def check_distance_collision(
    current_positions: List[List[float]], min_distance: float
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Check if any two agents are closer than minimum safe distance.

    Parameters:
    -----------
    current_positions : List[List[float]]
        List of [x, y] positions for each agent
    min_distance : float
        Minimum required distance between agents

    Returns:
    --------
    collision_detected : bool
        True if any pair is too close
    collision_pairs : List[Tuple[int, int]]
        List of (agent_i, agent_j) pairs that are too close
    """
    positions = np.array(current_positions)
    collision_pairs = []

    n_agents = len(positions)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance < min_distance:
                collision_pairs.append((i, j))

    collision_detected = len(collision_pairs) > 0
    return collision_detected, collision_pairs


def _minimum_distance_on_segments(
    start_a: np.ndarray,
    end_a: np.ndarray,
    start_b: np.ndarray,
    end_b: np.ndarray,
    t_min: float = 0.0,
    t_max: float = 1.0,
) -> Tuple[float, float]:
    """
    Compute the minimum distance between two linearly interpolated trajectories.

    Parameters:
    -----------
    start_a, end_a : np.ndarray
        Start and end points of first trajectory
    start_b, end_b : np.ndarray
        Start and end points of second trajectory
    t_min, t_max : float
        Time interval bounds within [0, 1] to evaluate

    Returns:
    --------
    min_distance : float
        Minimum Euclidean distance over t in [t_min, t_max]
    best_t : float
        Time at which the minimum distance occurs
    """
    rel_start = start_a - start_b
    rel_velocity = (end_a - start_a) - (end_b - start_b)
    velocity_norm_sq = float(np.dot(rel_velocity, rel_velocity))

    if velocity_norm_sq < 1e-12:
        candidate_times = [t_min, t_max]
    else:
        t_star = -float(np.dot(rel_start, rel_velocity)) / velocity_norm_sq
        t_star = float(np.clip(t_star, t_min, t_max))
        candidate_times = [t_min, t_max, t_star]

    min_distance = float("inf")
    best_t = t_min
    for t in candidate_times:
        delta = rel_start + t * rel_velocity
        distance = float(np.linalg.norm(delta))
        if distance < min_distance:
            min_distance = distance
            best_t = t

    return min_distance, best_t


def check_trajectory_collision_details(
    prev_positions: List[List[float]],
    current_positions: List[List[float]],
    min_distance: float,
    ignore_start: bool = True,
    start_epsilon: float = 1e-6,
) -> Tuple[bool, List[Tuple[int, int]], Dict[Tuple[int, int], float]]:
    """
    Check if any trajectories violate minimum separation during motion.

    Returns both collision pairs and the minimum distance observed per pair.
    """
    prev_pos = np.array(prev_positions, dtype=float)
    curr_pos = np.array(current_positions, dtype=float)
    collision_pairs = []
    pair_min_distances: Dict[Tuple[int, int], float] = {}

    t_min = start_epsilon if ignore_start else 0.0
    t_max = 1.0

    n_agents = len(prev_pos)
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            min_pair_distance, _ = _minimum_distance_on_segments(
                prev_pos[i], curr_pos[i], prev_pos[j], curr_pos[j], t_min=t_min, t_max=t_max
            )
            pair_min_distances[(i, j)] = min_pair_distance
            if min_pair_distance < min_distance:
                collision_pairs.append((i, j))

    collision_detected = len(collision_pairs) > 0
    return collision_detected, collision_pairs, pair_min_distances


def check_trajectory_collision(
    prev_positions: List[List[float]],
    current_positions: List[List[float]],
    min_distance: float,
    ignore_start: bool = True,
) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Backward-compatible wrapper for trajectory collision detection.

    Parameters:
    -----------
    prev_positions : List[List[float]]
        Previous [x, y] positions for each agent
    current_positions : List[List[float]]
        Current [x, y] positions for each agent
    min_distance : float
        Minimum required separation during motion
    ignore_start : bool
        If True, collisions at t=0 are ignored to avoid blocking recovery from
        already-unsafe initial states

    Returns:
    --------
    collision_detected : bool
        True if any trajectory pair comes too close
    collision_pairs : List[Tuple[int, int]]
        List of (agent_i, agent_j) pairs with unsafe trajectory separation
    """
    collision_detected, collision_pairs, _ = check_trajectory_collision_details(
        prev_positions, current_positions, min_distance, ignore_start=ignore_start
    )
    return collision_detected, collision_pairs


def check_collision(
    current_positions: List[List[float]],
    prev_positions: List[List[float]],
    min_distance: float,
) -> Tuple[bool, dict]:
    """
    Comprehensive collision detection: checks both distance and trajectory collisions.

    Parameters:
    -----------
    current_positions : List[List[float]]
        Current [x, y] positions for each agent
    prev_positions : List[List[float]]
        Previous [x, y] positions for each agent
    min_distance : float
        Minimum required distance between agents

    Returns:
    --------
    collision_detected : bool
        True if ANY collision type occurred (distance OR trajectory)
    collision_info : dict
        Detailed collision information with keys:
        - 'distance_collision': bool
        - 'trajectory_collision': bool
        - 'distance_pairs': List[Tuple[int, int]]
        - 'trajectory_pairs': List[Tuple[int, int]]
        - 'all_collision_pairs': List[Tuple[int, int]] (combined unique pairs)
    """
    # Check both collision types
    distance_collision, distance_pairs = check_distance_collision(
        current_positions, min_distance
    )
    trajectory_collision, trajectory_pairs, trajectory_min_distances = check_trajectory_collision_details(
        prev_positions, current_positions, min_distance
    )

    # Combine collision detection (OR logic)
    collision_detected = distance_collision or trajectory_collision

    # Get all unique collision pairs
    all_pairs = set(distance_pairs + trajectory_pairs)

    collision_info = {
        "distance_collision": distance_collision,
        "trajectory_collision": trajectory_collision,
        "distance_pairs": distance_pairs,
        "trajectory_pairs": trajectory_pairs,
        "trajectory_min_distances": {
            f"{i}-{j}": d for (i, j), d in trajectory_min_distances.items()
        },
        "all_collision_pairs": list(all_pairs),
    }

    return collision_detected, collision_info


def test_collision_detection():
    """Test the collision detection system with various scenarios."""
    print("=== Collision Detection Test Suite ===\n")

    # Test case 1: Distance collision
    current_pos_1 = [[0, 0], [1, 0], [5, 5]]  # Agents 0,1 are too close
    prev_pos_1 = [[0, 1], [1, 1], [4, 4]]
    min_dist = 3.0

    collision, info = check_collision(current_pos_1, prev_pos_1, min_dist)
    print("Test 1 - Distance Collision:")
    print(f"Collision detected: {collision}")
    print(f"Distance collision: {info['distance_collision']}")
    print(f"Distance pairs: {info['distance_pairs']}")
    print(f"All collision pairs: {info['all_collision_pairs']}")
    print()

    # Test case 2: Trajectory collision
    current_pos_2 = [[2, 0], [0, 2], [5, 5]]  # Agents 0,1 paths cross
    prev_pos_2 = [[0, 0], [2, 2], [4, 4]]

    collision, info = check_collision(current_pos_2, prev_pos_2, min_dist)
    print("Test 2 - Trajectory Collision:")
    print(f"Collision detected: {collision}")
    print(f"Trajectory collision: {info['trajectory_collision']}")
    print(f"Trajectory pairs: {info['trajectory_pairs']}")
    print(f"All collision pairs: {info['all_collision_pairs']}")
    print()

    # Test case 3: Both collision types
    current_pos_3 = [[0, 0], [1, 0], [4, 2]]  # Distance + trajectory collision
    prev_pos_3 = [[2, 2], [3, 2], [2, 4]]

    collision, info = check_collision(current_pos_3, prev_pos_3, min_dist)
    print("Test 3 - Both Collision Types:")
    print(f"Collision detected: {collision}")
    print(f"Distance collision: {info['distance_collision']}")
    print(f"Trajectory collision: {info['trajectory_collision']}")
    print(f"Distance pairs: {info['distance_pairs']}")
    print(f"Trajectory pairs: {info['trajectory_pairs']}")
    print(f"All collision pairs: {info['all_collision_pairs']}")
    print()

    # Test case 4: No collision
    current_pos_4 = [[0, 0], [10, 10], [20, 20]]
    prev_pos_4 = [[1, 1], [9, 9], [19, 19]]

    collision, info = check_collision(current_pos_4, prev_pos_4, min_dist)
    print("Test 4 - No Collision:")
    print(f"Collision detected: {collision}")
    print(f"Distance collision: {info['distance_collision']}")
    print(f"Trajectory collision: {info['trajectory_collision']}")
    print(f"All collision pairs: {info['all_collision_pairs']}")
    print()


if __name__ == "__main__":
    test_collision_detection()
