#!/usr/bin/env python3
"""
H2 Arm IK Solver Node
---------------------
Subscribes to:
  /target_pose_left  (geometry_msgs/PoseStamped) - desired left hand pose
  /target_pose_right (geometry_msgs/PoseStamped) - desired right hand pose

  All coordinates in PELVIS/WORLD frame (identical when pelvis is at world origin).

  Neutral hand positions:
    Left  hand: x=0.310, y=+0.183, z=0.214
    Right hand: x=0.310, y=-0.183, z=0.214

Publishes to:
  /joint_command (sensor_msgs/JointState) - arm joint angles in RADIANS for Isaac Sim

Key behaviours:
  - Left and right arms solved INDEPENDENTLY
  - Trajectory seeding: Cartesian waypoints prevent local minima
  - Background thread: IK never blocks the 50Hz publisher
  - Preemptable planning: new target cancels and replaces current plan
  - Reachability check: unreachable targets rejected, arm does not move
  - Velocity-limited smooth interpolation (MAX_JOINT_VEL deg/s)

Usage:
  conda activate ik_env
  source /opt/ros/jazzy/setup.bash
  python3 h2_ik_node.py
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from geometry_msgs.msg import PoseStamped

import pinocchio as pin
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
import time
import threading
import os

# ── Constants ──────────────────────────────────────────────────────────────────

_REPO_ROOT     = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
URDF_PATH      = os.path.join(_REPO_ROOT, "assets", "h2_description", "H2.urdf")
LEFT_EE_FRAME  = "left_hand_link"
RIGHT_EE_FRAME = "right_hand_link"

ARM_JOINT_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

LEFT_ARM_NAMES  = ARM_JOINT_NAMES[:7]
RIGHT_ARM_NAMES = ARM_JOINT_NAMES[7:]

PUBLISH_HZ      = 50
MAX_JOINT_VEL   = 30.0   # degrees per second
MIN_INTERP_TIME = 0.5    # seconds

IK_MAX_ITER             = 1000
IK_EPS_POS              = 1e-3    # 1mm position convergence
IK_EPS_ROT              = 1e-2    # rotation convergence
IK_DT                   = 0.5     # larger step for faster convergence
IK_DAMP                 = 1e-6
REACHABILITY_THRESH_M   = 0.05   # 30mm — reject if final error exceeds this
WAYPOINT_STEP_M         = 0.02   # 20mm Cartesian steps for trajectory seeding


class ArmState:
    """All mutable state for one arm, kept in one place for clarity."""

    def __init__(self, name, ee_id, v_indices, q_indices, neutral_q):
        self.name       = name          # "Left" or "Right"
        self.ee_id      = ee_id
        self.v_indices  = v_indices     # velocity indices in full model
        self.q_indices  = q_indices     # position indices in full model

        # Full 31-DoF q vector, warm-started for this arm's IK
        self.q_full     = neutral_q.copy()

        # 7-DoF arm joint angles (radians) — what gets published
        self.q_current  = np.zeros(7)
        self.q_target   = np.zeros(7)
        self.q_interp_start = np.zeros(7)

        self.interp_start   = None
        self.interp_dur     = 0.5

        # Threading
        self.lock           = threading.Lock()
        self.cancel_event   = threading.Event()
        self.plan_thread    = None


class H2IKNode(Node):

    def __init__(self):
        super().__init__("h2_ik_node")

        # ── Load Pinocchio model ───────────────────────────────────────────────
        self.model = pin.buildModelFromUrdf(URDF_PATH)
        self.get_logger().info(f"Loaded H2 URDF — DoFs: {self.model.nq}")

        left_ee_id  = self.model.getFrameId(LEFT_EE_FRAME)
        right_ee_id = self.model.getFrameId(RIGHT_EE_FRAME)

        def get_indices(names):
            v_idx, q_idx = [], []
            for name in names:
                jid = self.model.getJointId(name)
                if jid < self.model.njoints:
                    v_idx.append(self.model.joints[jid].idx_v)
                    q_idx.append(self.model.joints[jid].idx_q)
                else:
                    self.get_logger().warn(f"Joint not found: {name}")
            return v_idx, q_idx

        left_v,  left_q  = get_indices(LEFT_ARM_NAMES)
        right_v, right_q = get_indices(RIGHT_ARM_NAMES)

        neutral_q = pin.neutral(self.model)

        # Print neutral hand positions
        data = self.model.createData()
        pin.forwardKinematics(self.model, data, neutral_q)
        pin.updateFramePlacements(self.model, data)
        self.get_logger().info(
            f"Left  hand neutral: {data.oMf[left_ee_id].translation.round(3)}"
        )
        self.get_logger().info(
            f"Right hand neutral: {data.oMf[right_ee_id].translation.round(3)}"
        )

        # ── Arm state objects ─────────────────────────────────────────────────
        self.left  = ArmState("Left",  left_ee_id,  left_v,  left_q,  neutral_q)
        self.right = ArmState("Right", right_ee_id, right_v, right_q, neutral_q)

        # Initialization flag — node will not publish until joint states
        # are received from Isaac Sim, preventing snap-to-zero on startup
        self.initialized = False

        # ── ROS2 ──────────────────────────────────────────────────────────────
        self.pub = self.create_publisher(JointState, "joint_command", 10)

        # Read actual joint positions from Isaac Sim on startup
        self.create_subscription(
            JointState, "/joint_states", self.cb_init_joints, 10
        )
        self.create_subscription(
            PoseStamped, "/target_pose_left",  self.cb_left_target,  10
        )
        self.create_subscription(
            PoseStamped, "/target_pose_right", self.cb_right_target, 10
        )

        self.timer = self.create_timer(1.0 / PUBLISH_HZ, self.tick)
        self.get_logger().info(
            f"H2 IK node ready | {MAX_JOINT_VEL}°/s | "
            f"{PUBLISH_HZ}Hz → joint_command | "
            f"Waypoint step: {WAYPOINT_STEP_M*1000:.0f}mm | "
            f"Reject threshold: {REACHABILITY_THRESH_M*1000:.0f}mm"
        )
        self.get_logger().info(
            "Waiting for /joint_states from Isaac Sim before publishing..."
        )

    # ── Callbacks ─────────────────────────────────────────────────────────────

    def cb_init_joints(self, msg: JointState):
        """
        Read actual joint positions from Isaac Sim on first message.
        Prevents snap-to-zero when node starts while robot is mid-pose.
        Only runs once — ignored after initialization.
        """
        if self.initialized:
            return

        name_to_pos = dict(zip(msg.name, msg.position))

        for i, name in enumerate(LEFT_ARM_NAMES):
            if name in name_to_pos:
                self.left.q_current[i]      = name_to_pos[name]
                self.left.q_target[i]       = name_to_pos[name]
                self.left.q_interp_start[i] = name_to_pos[name]
                self.left.q_full[self.left.q_indices[i]] = name_to_pos[name]

        for i, name in enumerate(RIGHT_ARM_NAMES):
            if name in name_to_pos:
                self.right.q_current[i]      = name_to_pos[name]
                self.right.q_target[i]       = name_to_pos[name]
                self.right.q_interp_start[i] = name_to_pos[name]
                self.right.q_full[self.right.q_indices[i]] = name_to_pos[name]

        self.initialized = True
        self.get_logger().info(
            f"Initialized from Isaac Sim — "
            f"Left  arm (rad): {self.left.q_current.round(3).tolist()}"
        )
        self.get_logger().info(
            f"Initialized from Isaac Sim — "
            f"Right arm (rad): {self.right.q_current.round(3).tolist()}"
        )

    def cb_left_target(self, msg: PoseStamped):
        if not self.initialized:
            self.get_logger().warn("Not yet initialized — ignoring target")
            return
        self._dispatch(self.left, self._pose_msg_to_se3(msg))

    def cb_right_target(self, msg: PoseStamped):
        if not self.initialized:
            self.get_logger().warn("Not yet initialized — ignoring target")
            return
        self._dispatch(self.right, self._pose_msg_to_se3(msg))

    def _dispatch(self, arm: ArmState, target: pin.SE3):
        """Cancel any running plan and start a new one in a background thread."""
        # Signal the running thread to stop
        arm.cancel_event.set()
        if arm.plan_thread and arm.plan_thread.is_alive():
            arm.plan_thread.join(timeout=0.2)

        # Start fresh plan
        arm.cancel_event.clear()
        arm.plan_thread = threading.Thread(
            target=self._plan_and_solve,
            args=(arm, target),
            daemon=True
        )
        arm.plan_thread.start()

    # ── Trajectory seeding + IK (runs in background thread) ──────────────────

    def _plan_and_solve(self, arm: ArmState, target: pin.SE3):
        """
        1. Get current hand position via FK
        2. Generate Cartesian waypoints from current → target (20mm steps)
        3. Solve IK for each waypoint, warm-starting from previous solution
        4. Accept final solution only if within reachability threshold
        5. Trigger smooth interpolation on the main thread
        """
        # Create per-thread Pinocchio data (not thread-safe to share)
        data = self.model.createData()

        # Current q for this arm (snapshot, thread-safe copy)
        with arm.lock:
            q = arm.q_full.copy()
            # Sync q from actual current joint angles
            for i, idx in enumerate(arm.q_indices):
                q[idx] = arm.q_current[i]

        # Get current EE position via FK
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        current_pos = data.oMf[arm.ee_id].translation.copy()
        current_rot = data.oMf[arm.ee_id].rotation.copy()

        target_pos  = target.translation
        target_rot  = target.rotation

        # Generate Cartesian waypoints
        dist        = np.linalg.norm(target_pos - current_pos)
        n_waypoints = max(int(np.ceil(dist / WAYPOINT_STEP_M)), 1)

        self.get_logger().info(
            f"{arm.name}: planning {n_waypoints} waypoints "
            f"over {dist*1000:.0f}mm"
        )

        # Interpolate positions linearly, rotations via Slerp
        positions = [
            current_pos + (target_pos - current_pos) * (i / n_waypoints)
            for i in range(1, n_waypoints + 1)
        ]

        # Slerp between current and target rotation
        r_current = Rotation.from_matrix(current_rot)
        r_target  = Rotation.from_matrix(target_rot)
        key_rots  = Rotation.concatenate([r_current, r_target])
        slerp     = Slerp([0, 1], key_rots)
        rotations = [
            slerp(i / n_waypoints).as_matrix()
            for i in range(1, n_waypoints + 1)
        ]

        # Solve IK for each waypoint sequentially
        for wp_idx, (pos, rot) in enumerate(zip(positions, rotations)):

            # Check for cancellation (new target arrived)
            if arm.cancel_event.is_set():
                self.get_logger().info(
                    f"{arm.name}: plan cancelled at waypoint "
                    f"{wp_idx+1}/{n_waypoints}"
                )
                return

            waypoint = pin.SE3(rot, pos)

            for iteration in range(IK_MAX_ITER):
                if arm.cancel_event.is_set():
                    return

                pin.forwardKinematics(self.model, data, q)
                pin.updateFramePlacements(self.model, data)

                oMf = data.oMf[arm.ee_id]
                err = pin.log6(oMf.inverse() * waypoint).vector
                J   = pin.computeFrameJacobian(
                    self.model, data, q,
                    arm.ee_id,
                    pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
                )

                if np.linalg.norm(err[:3]) < IK_EPS_POS and np.linalg.norm(err[3:]) < IK_EPS_ROT:
                    break

                JJT  = J @ J.T
                damp = IK_DAMP * np.eye(JJT.shape[0])
                dq   = J.T @ np.linalg.solve(JJT + damp, err)

                dq_masked = np.zeros(self.model.nv)
                for idx in arm.v_indices:
                    dq_masked[idx] = dq[idx]

                q = pin.integrate(self.model, q, IK_DT * dq_masked)
                q = np.clip(
                    q,
                    self.model.lowerPositionLimit,
                    self.model.upperPositionLimit
                )

        # Final reachability check on the last waypoint (= target)
        pin.forwardKinematics(self.model, data, q)
        pin.updateFramePlacements(self.model, data)
        achieved = data.oMf[arm.ee_id].translation
        err_m    = np.linalg.norm(achieved - target_pos)
        err_mm   = err_m * 1000

        self.get_logger().info(
            f"{arm.name} final error: {err_mm:.1f}mm | "
            f"achieved: {achieved.round(3)} | desired: {target_pos.round(3)}"
        )

        if err_m > REACHABILITY_THRESH_M:
            self.get_logger().warn(
                f"{arm.name} target REJECTED — {err_mm:.1f}mm exceeds "
                f"{REACHABILITY_THRESH_M*1000:.0f}mm threshold. Arm will NOT move."
            )
            return

        # Extract final arm joint angles
        new_arm_q = np.array([q[idx] for idx in arm.q_indices])

        # Velocity-limited travel time
        with arm.lock:
            delta_deg    = np.abs(np.degrees(new_arm_q) - np.degrees(arm.q_current))
            max_delta    = np.max(delta_deg)
            interp_dur   = max(max_delta / MAX_JOINT_VEL, MIN_INTERP_TIME)

            self.get_logger().info(
                f"{arm.name} solution (deg): "
                f"{np.degrees(new_arm_q).round(1).tolist()}"
            )
            self.get_logger().info(
                f"{arm.name} max delta: {max_delta:.1f}° → "
                f"travel time: {interp_dur:.2f}s at {MAX_JOINT_VEL}°/s"
            )

            # Commit new target — picked up by tick() on next cycle
            arm.q_full          = q
            arm.q_interp_start  = arm.q_current.copy()
            arm.q_target        = new_arm_q
            arm.interp_start    = time.time()
            arm.interp_dur      = interp_dur

    # ── Publish tick (main thread, 50Hz) ──────────────────────────────────────

    def tick(self):
        # Don't publish until initialized from Isaac Sim joint states
        if not self.initialized:
            return

        now = time.time()

        for arm in (self.left, self.right):
            with arm.lock:
                if arm.interp_start is not None:
                    elapsed = now - arm.interp_start
                    alpha   = min(elapsed / arm.interp_dur, 1.0)
                    alpha_s = alpha * alpha * (3.0 - 2.0 * alpha)
                    arm.q_current = (
                        arm.q_interp_start +
                        alpha_s * (arm.q_target - arm.q_interp_start)
                    )
                    if alpha >= 1.0:
                        arm.interp_start = None

        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name         = ARM_JOINT_NAMES
        msg.position     = (
            self.left.q_current.tolist() +
            self.right.q_current.tolist()
        )
        self.pub.publish(msg)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _pose_msg_to_se3(self, msg: PoseStamped) -> pin.SE3:
        p = msg.pose.position
        o = msg.pose.orientation
        t = np.array([p.x, p.y, p.z])
        R = Rotation.from_quat([o.x, o.y, o.z, o.w]).as_matrix()
        return pin.SE3(R, t)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    rclpy.init()
    node = H2IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        try:
            rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
