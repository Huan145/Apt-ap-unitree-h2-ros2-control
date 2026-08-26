#!/usr/bin/env python3
"""
Generate offline IK trajectory library for H2 5-phase grasp+lift sequence.

Updated 5-Phase Logic:
  1. Wide open stance (X=0.15m, free orientation)
  2. Reach to grasp standoff position (Full SE3)
  3. Approach (pinch inward)
  4. Wrist tilt (applied immediately after pinch approach)
  5. Carry/lift pose
"""

import os
import time
import numpy as np
import torch
import pinocchio as pin
from scipy.spatial.transform import Rotation, Slerp

# ── Paths & File Settings ───────────────────────────────────────────────────

URDF_PATH = os.path.expanduser("~/Github/h2-ros2-control/assets/h2_description/H2.urdf")
OUTPUT_FILE = "ik_trajectory_library_5phase.pt"

# ── Robot & Trajectory Configurations ────────────────────────────────────────

N_TIMESTEPS = 1000

LEFT_ARM_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

RIGHT_ARM_NAMES = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# Exact Standing / Relaxed Configurations extracted from Isaac Sim
RELAXED_LEFT_ARM = np.array([0.2802, 0.1350, -0.0088, 0.5610, 0.0004, 0.0107, -0.0006])
RELAXED_RIGHT_ARM = np.array([0.2802, -0.1355, 0.0089, 0.5611, -0.0004, 0.0107, 0.0006])

# IK Solver Hyperparameters
IK_MAX_ITER = 5000
IK_EPS_POS = 1e-2     # 10 mm tolerance
IK_EPS_ROT = 1e-1     # ~5.7 deg tolerance
IK_DT = 0.1
IK_DAMP = 1e-5

# Box & Geometry Parameters (meters)
BOX_HALF_X = 0.20
BOX_HALF_Y = 0.15
BOX_HALF_Z = 0.075

WIDE_STANDOFF_OFFSET_Y = 0.2  # Outward offset for wide stance
PHASE1_TARGET_X = 0.05         # Explicit X position for Phase 1 hands
GRASP_REACH = 0.05
GRASP_PELVIS_OFFSET = np.array([-0.09, 0.0, 0.02])
APPROACH_DIST_M = 0.07

# Carry Pose (Phase 5)
LIFT_X = 0.20
LIFT_HALF_WIDTH_Y = 0.20
LIFT_Z = 0.275

# Wrist Tilt (Phase 4)
WRIST_PITCH_LOCAL_IDX = 5
WRIST_TILT_DEG = -9.0

# Grid Definition (5x5x5x5 = 625 target poses)
X_RANGE = np.linspace(0.35, 0.55, 5)
Y_RANGE = np.linspace(-0.20, 0.20, 5)
Z_RANGE = np.linspace(0.18, 0.30, 5)
YAW_RANGE = np.linspace(-20, 20, 5) * np.pi / 180


# ── Flexible IK Solver ───────────────────────────────────────────────────────

def solve_ik(model, data, arm_names, ee_id, q_warm_start, target_pos, target_rot=None):
    q = q_warm_start.copy()
    
    for iteration in range(IK_MAX_ITER):
        pin.forwardKinematics(model, data, q)
        pin.updateFramePlacements(model, data)
        
        oMf = data.oMf[ee_id]
        current_pos = oMf.translation
        
        if target_rot is not None:
            # Full SE(3) tracking
            target_se3 = pin.SE3(target_rot, target_pos)
            err = pin.log6(oMf.inverse() * target_se3).vector
            pos_err = np.linalg.norm(err[:3])
            rot_err = np.linalg.norm(err[3:])
            
            if pos_err < IK_EPS_POS and rot_err < IK_EPS_ROT:
                arm_q = np.array([q[model.joints[model.getJointId(n)].idx_q] for n in arm_names])
                return arm_q, q
                
            J = pin.computeFrameJacobian(
                model, data, q, ee_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
        else:
            # Position-only tracking (free orientation)
            pos_err_vec = target_pos - current_pos
            pos_err = np.linalg.norm(pos_err_vec)
            
            if pos_err < IK_EPS_POS:
                arm_q = np.array([q[model.joints[model.getJointId(n)].idx_q] for n in arm_names])
                return arm_q, q
                
            err = np.zeros(6)
            err[:3] = pos_err_vec
            
            J_full = pin.computeFrameJacobian(
                model, data, q, ee_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
            )
            J = J_full.copy()
            J[3:, :] = 0.0
        
        JJT = J @ J.T
        damp = IK_DAMP * np.eye(JJT.shape[0])
        dq = J.T @ np.linalg.solve(JJT + damp, err)
        
        dq_masked = np.zeros(model.nv)
        for arm_name in arm_names:
            joint_id = model.getJointId(arm_name)
            v_idx = model.joints[joint_id].idx_v
            dq_masked[v_idx] = dq[v_idx]
            
        q = pin.integrate(model, q, IK_DT * dq_masked)
        q = np.clip(q, model.lowerPositionLimit, model.upperPositionLimit)
        
    return None, q


# ── 5-Phase Trajectory Generator ─────────────────────────────────────────────

def generate_full_trajectory(model, data, arm_names, ee_id, q_relaxed_full, 
                            box_center, box_rotation):
    is_left = (ee_id == model.getFrameId("left_hand_link"))
    
    trajectory = []
    
    # Get current FK placement from relaxed initial seed
    pin.forwardKinematics(model, data, q_relaxed_full)
    pin.updateFramePlacements(model, data)
    initial_pos = data.oMf[ee_id].translation.copy()
    initial_rot = data.oMf[ee_id].rotation.copy()
    
    # Compute base grasp point (Phase 2)
    box_y_axis = box_rotation[:, 1]
    if is_left:
        phase2_grasp_pos = box_center + (BOX_HALF_Y + GRASP_REACH) * box_y_axis + GRASP_PELVIS_OFFSET
    else:
        phase2_grasp_pos = box_center - (BOX_HALF_Y + GRASP_REACH) * box_y_axis + GRASP_PELVIS_OFFSET
        
    # Phase 1: Wide stance
    phase1_wide_pos = phase2_grasp_pos.copy()
    phase1_wide_pos[0] = PHASE1_TARGET_X
    
    if is_left:
        phase1_wide_pos[1] += WIDE_STANDOFF_OFFSET_Y
    else:
        phase1_wide_pos[1] -= WIDE_STANDOFF_OFFSET_Y
        
    # Phase 3: Approach point (pinch inward)
    phase3_approach_pos = phase2_grasp_pos.copy()
    if is_left:
        phase3_approach_pos[1] -= APPROACH_DIST_M
    else:
        phase3_approach_pos[1] += APPROACH_DIST_M
        
    # Phase 5: Carry pose
    if is_left:
        phase5_carry_pos = np.array([LIFT_X, LIFT_HALF_WIDTH_Y, LIFT_Z])
    else:
        phase5_carry_pos = np.array([LIFT_X, -LIFT_HALF_WIDTH_Y, LIFT_Z])
        
    R_identity = np.eye(3)
    
    # ── Segment A: Reach & Pinch (Phases 1-3) ──
    waypoints_reach = [
        (phase1_wide_pos, None),            # Phase 1: Wide stance
        (phase2_grasp_pos, R_identity),     # Phase 2: Move to standoff
        (phase3_approach_pos, R_identity),  # Phase 3: Pinch approach
    ]
    
    q = q_relaxed_full.copy()
    current_pos = initial_pos
    current_rot = initial_rot
    
    for wp_pos, wp_rot in waypoints_reach:
        prev_pos = current_pos
        prev_rot = current_rot
        
        dist = np.linalg.norm(wp_pos - prev_pos)
        n_steps = max(int(np.ceil(dist / 0.02)), 2)
        
        positions = [
            prev_pos + (wp_pos - prev_pos) * (i / n_steps)
            for i in range(1, n_steps + 1)
        ]
        
        if wp_rot is not None:
            r_prev = Rotation.from_matrix(prev_rot)
            r_target = Rotation.from_matrix(wp_rot)
            slerp = Slerp([0, 1], Rotation.concatenate([r_prev, r_target]))
            rotations = [slerp(i / n_steps).as_matrix() for i in range(1, n_steps + 1)]
        else:
            rotations = [None] * n_steps
        
        for pos, rot in zip(positions, rotations):
            q_sol, q = solve_ik(model, data, arm_names, ee_id, q, pos, rot)
            if q_sol is None:
                return None
            trajectory.append(q_sol)
            
        current_pos = wp_pos
        if wp_rot is not None:
            current_rot = wp_rot
        else:
            pin.forwardKinematics(model, data, q)
            pin.updateFramePlacements(model, data)
            current_rot = data.oMf[ee_id].rotation.copy()

    # ── Phase 4: Wrist Tilt (Applied directly after Pinch) ──
    tilt_steps = 6
    start_q_tilt = trajectory[-1].copy()
    target_q_tilt = start_q_tilt.copy()
    
    target_q_tilt[WRIST_PITCH_LOCAL_IDX] += np.radians(WRIST_TILT_DEG)
    wrist_joint = arm_names[WRIST_PITCH_LOCAL_IDX]
    wrist_idx = model.joints[model.getJointId(wrist_joint)].idx_q
    target_q_tilt[WRIST_PITCH_LOCAL_IDX] = np.clip(
        target_q_tilt[WRIST_PITCH_LOCAL_IDX],
        model.lowerPositionLimit[wrist_idx],
        model.upperPositionLimit[wrist_idx]
    )
    
    for i in range(1, tilt_steps + 1):
        alpha = i / tilt_steps
        q_interp = start_q_tilt + alpha * (target_q_tilt - start_q_tilt)
        trajectory.append(q_interp)

    # Run Forward Kinematics on the final tilted joint state
    # to extract the TRUE tilted 3D rotation matrix
    q_tilted_full = q.copy()
    for i, name in enumerate(arm_names):
        jid = model.getJointId(name)
        q_tilted_full[model.joints[jid].idx_q] = target_q_tilt[i]

    pin.forwardKinematics(model, data, q_tilted_full)
    pin.updateFramePlacements(model, data)
    tilted_rot = data.oMf[ee_id].rotation.copy()  # Holds the true -9 deg tilted orientation

    # ── Segment B: Carry Pose (Phase 5) ──
    carry_steps = 15
    start_pos = current_pos.copy()
    target_lift_rot = tilted_rot  # Enforce the newly computed tilted rotation matrix

    for i in range(1, carry_steps + 1):
        alpha = i / carry_steps
        interp_pos = start_pos + alpha * (phase5_carry_pos - start_pos)
        
        # IK now tracks the lift while holding the true tilted orientation
        q_sol_carry, q = solve_ik(model, data, arm_names, ee_id, q, interp_pos, target_lift_rot)
        if q_sol_carry is None:
            return None
        trajectory.append(q_sol_carry)

    # ── Resample trajectory to fixed N_TIMESTEPS grid ──
    trajectory_arr = np.array(trajectory)
    if len(trajectory_arr) < N_TIMESTEPS:
        t_old = np.linspace(0, 1, len(trajectory_arr))
        t_new = np.linspace(0, 1, N_TIMESTEPS)
        from scipy.interpolate import interp1d
        trajectory_interp = interp1d(t_old, trajectory_arr, axis=0)(t_new)
    else:
        indices = np.linspace(0, len(trajectory_arr) - 1, N_TIMESTEPS, dtype=int)
        trajectory_interp = trajectory_arr[indices, :]

    return trajectory_interp

# ── Main Execution ───────────────────────────────────────────────────────────

def main():
    print(f"Loading URDF: {URDF_PATH}")
    if not os.path.exists(URDF_PATH):
        raise FileNotFoundError(f"URDF file not found at '{URDF_PATH}'. Please check the path.")

    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    
    left_ee_id = model.getFrameId("left_hand_link")
    right_ee_id = model.getFrameId("right_hand_link")
    
    # Initialize full model state with relaxed standing posture
    q_relaxed_full = pin.neutral(model)
    for i, name in enumerate(LEFT_ARM_NAMES):
        jid = model.getJointId(name)
        q_relaxed_full[model.joints[jid].idx_q] = RELAXED_LEFT_ARM[i]
        
    for i, name in enumerate(RIGHT_ARM_NAMES):
        jid = model.getJointId(name)
        q_relaxed_full[model.joints[jid].idx_q] = RELAXED_RIGHT_ARM[i]
        
    n_x, n_y, n_z, n_yaw = len(X_RANGE), len(Y_RANGE), len(Z_RANGE), len(YAW_RANGE)
    total = n_x * n_y * n_z * n_yaw
    
    left_table = torch.full((n_x, n_y, n_z, n_yaw, N_TIMESTEPS, 7), float('nan'))
    right_table = torch.full((n_x, n_y, n_z, n_yaw, N_TIMESTEPS, 7), float('nan'))
    
    success_count = 0
    failed_poses = []
    
    print(f"\nGenerating {total} 5-phase trajectory pairs ({n_x}x{n_y}x{n_z}x{n_yaw})...")
    print(f"  Phase 1: Wide stance (X = {PHASE1_TARGET_X:.2f}m, Y offset = {WIDE_STANDOFF_OFFSET_Y:.2f}m)")
    print(f"  Phase 2: Grasp standoff reach")
    print(f"  Phase 3: Approach pinch")
    print(f"  Phase 4: Wrist tilt ({WRIST_TILT_DEG}°)")
    print(f"  Phase 5: Carry pose lift\n")
    
    t0 = time.time()
    
    for i_x, x in enumerate(X_RANGE):
        for i_y, y in enumerate(Y_RANGE):
            for i_z, z in enumerate(Z_RANGE):
                for i_yaw, yaw in enumerate(YAW_RANGE):
                    box_center = np.array([x, y, z])
                    box_rotation = Rotation.from_euler("z", yaw).as_matrix()
                    
                    left_traj = generate_full_trajectory(
                        model, data, LEFT_ARM_NAMES, left_ee_id,
                        q_relaxed_full, box_center, box_rotation
                    )
                    
                    right_traj = generate_full_trajectory(
                        model, data, RIGHT_ARM_NAMES, right_ee_id,
                        q_relaxed_full, box_center, box_rotation
                    )
                    
                    if left_traj is not None and right_traj is not None:
                        left_table[i_x, i_y, i_z, i_yaw] = torch.tensor(left_traj, dtype=torch.float32)
                        right_table[i_x, i_y, i_z, i_yaw] = torch.tensor(right_traj, dtype=torch.float32)
                        success_count += 1
                    else:
                        failed_poses.append((x, y, z, np.degrees(yaw)))
                        
                    idx = i_x * n_y * n_z * n_yaw + i_y * n_z * n_yaw + i_z * n_yaw + i_yaw
                    if (idx + 1) % 50 == 0 or idx == total - 1:
                        elapsed = time.time() - t0
                        rate = (idx + 1) / elapsed if elapsed > 0 else 0
                        eta = (total - idx - 1) / rate if rate > 0 else 0
                        print(
                            f"  [{idx+1:4d}/{total}] "
                            f"({success_count:4d} success, {len(failed_poses):3d} failed) "
                            f"{rate:.1f} traj/s, ETA {eta:.0f}s"
                        )

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"Generated {success_count}/{total} successful trajectory pairs in {elapsed:.1f}s")
    print(f"Success rate: {100 * success_count / total:.1f}%")

    torch.save({
        "left": left_table,
        "right": right_table,
        "x_range": X_RANGE,
        "y_range": Y_RANGE,
        "z_range": Z_RANGE,
        "yaw_range": YAW_RANGE,
        "box_half_dims": np.array([BOX_HALF_X, BOX_HALF_Y, BOX_HALF_Z]),
        "grasp_reach": GRASP_REACH,
        "grasp_offset": GRASP_PELVIS_OFFSET,
        "approach_dist": APPROACH_DIST_M,
        "lift_pose": np.array([LIFT_X, LIFT_HALF_WIDTH_Y, LIFT_Z]),
        "wrist_tilt_deg": WRIST_TILT_DEG,
        "success_rate": success_count / total,
    }, OUTPUT_FILE)
    
    print(f"\nSaved trajectory library to '{OUTPUT_FILE}'")


if __name__ == "__main__":
    main()