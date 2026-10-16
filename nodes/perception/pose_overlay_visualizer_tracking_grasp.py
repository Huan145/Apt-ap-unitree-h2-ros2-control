#!/usr/bin/env python3
"""
Pose overlay visualizer for the TRACKING pipeline.
Subscribes to /tracking/output (~20Hz) instead of /output (~1.5Hz).
Box dimensions read dynamically from bbox.size.
Camera intrinsics read dynamically from /camera_info.

Grasp orientation convention:
  The H2 hand Y axis is the palm normal (at neutral: palm faces -Y pelvis for left hand).
  Target orientation aligns hand Y with the box face normal, so the palm is always
  perpendicular to the box Y face regardless of box rotation.

  Left  hand (G1, +Y face): hand Y must point in -box_Y_pelvis direction
  Right hand (G2, -Y face): hand Y must point in +box_Y_pelvis direction

  Fingers (+X) are kept as close to +X pelvis as possible.
  This is computed via the rotation-between-vectors approach (no hardcoded matrices).

Coordinate frames:
  Camera frame : x=right, y=down,    z=forward
  Pelvis frame : x=forward, y=left,  z=up
  Box frame    : defined by FoundationPose mesh origin

Subscribes:
  /tracking/output  - vision_msgs/Detection3DArray
  /rgb              - sensor_msgs/Image
  /camera_info      - sensor_msgs/CameraInfo
Publishes:
  /tracking_visualization - sensor_msgs/Image
  /target_pose_left       - geometry_msgs/PoseStamped (pelvis frame)
  /target_pose_right      - geometry_msgs/PoseStamped (pelvis frame)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from vision_msgs.msg import Detection3DArray
from geometry_msgs.msg import PoseStamped
import tf2_ros
import numpy as np
import cv2
import time

AXIS_LENGTH  = 0.12
GRASP_REACH  = 0.05   # metres outside the box Y face

# Extra fixed offset applied to both grasp points AFTER transforming to
# pelvis frame (x=forward, y=left, z=up).
#   x: negative = closer to the robot/camera (less forward reach)
#   z: negative = lower
GRASP_PELVIS_OFFSET = np.array([-0.09, 0.0, 0.02])   # (forward, left, up) metres

PELVIS_FRAME = 'pelvis'
CAMERA_FRAME = 'Camera'

BOX_EDGES = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7),
]


# ── Geometry helpers ──────────────────────────────────────────────────────────

def get_box_corners(hx, hy, hz):
    return np.array([
        [ hx,  hy,  hz], [ hx, -hy,  hz],
        [-hx, -hy,  hz], [-hx,  hy,  hz],
        [ hx,  hy, -hz], [ hx, -hy, -hz],
        [-hx, -hy, -hz], [-hx,  hy, -hz],
    ], dtype=np.float64)

def get_axes(length):
    return np.array([
        [0., 0., 0.], [length, 0., 0.],
        [0., length, 0.], [0., 0., length],
    ], dtype=np.float64)

def quat_to_rot(q):
    """Quaternion (x,y,z,w) → 3×3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1-2*(y*y+z*z),  2*(x*y-z*w),  2*(x*z+y*w)],
        [  2*(x*y+z*w),1-2*(x*x+z*z),  2*(y*z-x*w)],
        [  2*(x*z-y*w),  2*(y*z+x*w),1-2*(x*x+y*y)],
    ], dtype=np.float64)

def rot_to_quat(R):
    """3×3 rotation matrix → quaternion (x,y,z,w)."""
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])

def project(points_3d, R, t, fx, fy, cx, cy):
    cam = (R @ points_3d.T).T + t
    pts = np.full((len(cam), 2), -1.0)
    for i, p in enumerate(cam):
        if p[2] > 0:
            pts[i, 0] = fx * p[0] / p[2] + cx
            pts[i, 1] = fy * p[1] / p[2] + cy
    return pts.astype(np.float32)

def transform_stamped_to_matrix(ts):
    tr = ts.transform.translation
    ro = ts.transform.rotation
    R  = quat_to_rot((ro.x, ro.y, ro.z, ro.w))
    T  = np.eye(4)
    T[:3, :3] = R
    T[:3,  3] = [tr.x, tr.y, tr.z]
    return T

def obj_to_cam(pt_obj, R_box_cam, t_box_cam):
    return R_box_cam @ pt_obj + t_box_cam

def cam_to_pelvis(pt_cam, T_pelvis_cam):
    return (T_pelvis_cam @ np.array([*pt_cam, 1.0]))[:3]

def rotation_from_y_axis(target_y, reference_x=np.array([1., 0., 0.])):
    """
    Build a rotation matrix whose Y column = target_y (normalised),
    with X column as close to reference_x as possible.

    This gives the hand orientation where:
      - hand Y axis (palm normal) faces target_y direction
      - hand X axis (fingers)    stays as close to +X pelvis as possible
      - hand Z axis (thumb)      points upward (+Z pelvis)

    The Z-up check ensures the thumb always points up regardless of
    which side of the box the hand is on.

    Args:
      target_y    : desired hand Y axis direction in pelvis frame (will be normalised)
      reference_x : preferred finger direction (default +X pelvis = forward)

    Returns: 3×3 rotation matrix
    """
    y = target_y / np.linalg.norm(target_y)

    # X = reference_x orthogonalised against y (Gram-Schmidt)
    x = reference_x - np.dot(reference_x, y) * y
    x_norm = np.linalg.norm(x)
    if x_norm < 1e-6:
        # reference_x is parallel to y — pick a fallback
        x = np.array([0., 0., 1.])
        x = x - np.dot(x, y) * y
        x_norm = np.linalg.norm(x)
    x = x / x_norm

    # Z completes right-handed frame: Z = X × Y
    z = np.cross(x, y)

    # Ensure Z points generally upward (+Z pelvis = thumb up)
    # If Z points down, flip both X and Z — frame stays right-handed
    # because negating two axes preserves determinant = +1
    if z[2] < 0:
        x = -x
        z = -z

    # Columns of R are the hand axes expressed in pelvis frame
    R = np.column_stack([x, y, z])
    return R

def compute_grasp_quaternions(R_box_cam, T_pelvis_cam):
    """
    Compute left and right hand grasp quaternions in pelvis frame.

    Both hands get the SAME orientation: hand Y = box Y axis (sign flipped to
    point left), fingers as close to +X pelvis as possible. For a box square to
    the robot that is identity for both hands - what the old code produced
    via its |w| < 0.1 fallback for the left hand; the old left formula
    (hand Y -> -box_Y) is a 180 deg turn (fingers backward) that the fallback
    only caught for box yaw < ~11 deg. FoundationPose may also flip the box Y
    sign between frames (symmetric box); normalising the sign removes that.

    Fingers (+X) are kept pointing as close to +X pelvis (forward) as possible.

    Args:
      R_box_cam    : 3×3 rotation of box in camera frame (from FoundationPose)
      T_pelvis_cam : 4×4 transform from camera to pelvis (from TF)

    Returns: (q_left, q_right) each as (x,y,z,w) numpy array
    """
    R_pelvis_cam = T_pelvis_cam[:3, :3]

    # Box Y axis in camera frame (col 1 of box rotation matrix)
    box_y_cam = R_box_cam[:, 1]

    # Box Y axis in pelvis frame
    box_y_pelvis = R_pelvis_cam @ box_y_cam
    box_y_pelvis = box_y_pelvis / np.linalg.norm(box_y_pelvis)

    if box_y_pelvis[1] < 0:              # box Y sign is arbitrary -> make it point left
        box_y_pelvis = -box_y_pelvis
    q = rot_to_quat(rotation_from_y_axis(box_y_pelvis))
    return q, q


# ── Overlay drawing ───────────────────────────────────────────────────────────

def draw_overlay(img_bgr, pose, box_size, fx, fy, cx, cy):
    p = pose.position
    q = pose.orientation
    R = quat_to_rot((q.x, q.y, q.z, q.w))
    t = np.array([p.x, p.y, p.z])

    hx = box_size.x / 2.0
    hy = box_size.y / 2.0
    hz = box_size.z / 2.0

    corners = project(get_box_corners(hx, hy, hz), R, t, fx, fy, cx, cy)
    axes    = project(get_axes(AXIS_LENGTH), R, t, fx, fy, cx, cy)

    # Wireframe (cyan)
    for i, j in BOX_EDGES:
        if corners[i,0] < 0 or corners[j,0] < 0:
            continue
        cv2.line(img_bgr,
                 tuple(corners[i].astype(int)),
                 tuple(corners[j].astype(int)),
                 (255, 255, 0), 2)

    # Axes
    if axes[0,0] >= 0:
        orig = tuple(axes[0].astype(int))
        cv2.arrowedLine(img_bgr, orig, tuple(axes[1].astype(int)),
                        (0,   0, 255), 2, tipLength=0.2)  # X red
        cv2.arrowedLine(img_bgr, orig, tuple(axes[2].astype(int)),
                        (0, 255,   0), 2, tipLength=0.2)  # Y green
        cv2.arrowedLine(img_bgr, orig, tuple(axes[3].astype(int)),
                        (255,   0,   0), 2, tipLength=0.2)  # Z blue

    cv2.putText(img_bgr,
                f"[TRACKING] x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}m",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,0), 2)
    cv2.putText(img_bgr,
                f"box: {box_size.x:.2f}x{box_size.y:.2f}x{box_size.z:.2f}m",
                (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1)
    cv2.putText(img_bgr,
                f"cam: fx={fx:.1f} cx={cx:.1f} cy={cy:.1f}",
                (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)

    # Grasp points in object frame (on the box ±Y faces, standoff = GRASP_REACH)
    g1_obj = np.array([0.0,  hy + GRASP_REACH, 0.0])
    g2_obj = np.array([0.0, -(hy + GRASP_REACH), 0.0])

    # Transform to camera frame (raw box-face points, no offset here —
    # the closer/lower nudge is applied later in PELVIS frame, see
    # GRASP_PELVIS_OFFSET in rgb_callback)
    g1_cam = obj_to_cam(g1_obj, R, t)
    g2_cam = obj_to_cam(g2_obj, R, t)

    grasp_px = np.full((2, 2), -1.0)
    for i, p_cam in enumerate([g1_cam, g2_cam]):
        if p_cam[2] > 0:
            grasp_px[i, 0] = fx * p_cam[0] / p_cam[2] + cx
            grasp_px[i, 1] = fy * p_cam[1] / p_cam[2] + cy

    for (label, color), (gx, gy) in zip(
            [('G1',(0,255,255)), ('G2',(0,120,255))], grasp_px):
        if gx < 0:
            continue
        cxi, cyi = int(gx), int(gy)
        h = 14
        cv2.line(img_bgr, (cxi-h,cyi-h), (cxi+h,cyi+h), color, 3, cv2.LINE_AA)
        cv2.line(img_bgr, (cxi+h,cyi-h), (cxi-h,cyi+h), color, 3, cv2.LINE_AA)
        cv2.circle(img_bgr, (cxi,cyi), 4, color, -1, cv2.LINE_AA)
        cv2.putText(img_bgr, label, (cxi+h+4, cyi-h+4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)

    return img_bgr, g1_cam, g2_cam, R   # R = R_box_cam


# ── ROS node ──────────────────────────────────────────────────────────────────

class PoseOverlayVisualizerTracking(Node):

    def __init__(self):
        super().__init__('pose_overlay_visualizer_tracking')

        self.latest_pose       = None
        self.latest_stamp      = None   # image time of the detection (TF is looked up at this time)
        self.latest_box_size   = None
        self.last_publish_time = 0.0
        self.MIN_INTERVAL      = 0.05   # 20 Hz cap

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.tf_buffer   = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self.create_subscription(
            Detection3DArray, '/tracking/output', self.pose_callback, 10)
        self.create_subscription(
            Image, '/rgb', self.rgb_callback, 10)
        self.create_subscription(
            CameraInfo, '/camera_info', self.camera_info_callback, 10)

        self.pub_vis   = self.create_publisher(Image,       '/tracking_visualization', 10)
        self.pub_left  = self.create_publisher(PoseStamped, '/target_pose_left',       10)
        self.pub_right = self.create_publisher(PoseStamped, '/target_pose_right',      10)

        self.get_logger().info(
            'PoseOverlayVisualizerTracking ready.\n'
            f'  TF chain     : {CAMERA_FRAME} → {PELVIS_FRAME}\n'
            '  Subscribing  : /tracking/output  /rgb  /camera_info\n'
            '  Publishing   : /tracking_visualization  '
            '/target_pose_left  /target_pose_right\n'
            '  Grasp orient : hand Y axis aligned with box face normal\n'
            '    Left  (G1) : hand Y → -box_Y_pelvis\n'
            '    Right (G2) : hand Y → +box_Y_pelvis\n'
            '  Intrinsics   : waiting for /camera_info...'
        )

    def _get_T_pelvis_cam(self, stamp=None):
        """Camera pose in the pelvis frame at `stamp` (the image the box pose came
        from); falls back to the latest TF if that time is not in the buffer."""
        try:
            if stamp is not None:
                try:
                    ts = self.tf_buffer.lookup_transform(
                        PELVIS_FRAME, CAMERA_FRAME, rclpy.time.Time.from_msg(stamp),
                        timeout=rclpy.duration.Duration(seconds=0.1))
                    return transform_stamped_to_matrix(ts)
                except tf2_ros.ExtrapolationException:
                    self.get_logger().warn('TF at detection time not available - using latest',
                                           throttle_duration_sec=5.0)
            ts = self.tf_buffer.lookup_transform(
                PELVIS_FRAME, CAMERA_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.1),
            )
            return transform_stamped_to_matrix(ts)
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            self.get_logger().warn(
                f'TF lookup failed ({PELVIS_FRAME}←{CAMERA_FRAME}): {e}',
                throttle_duration_sec=2.0,
            )
            return None

    def _make_pose_stamped(self, pos, quat, stamp):
        msg = PoseStamped()
        msg.header.stamp    = stamp
        msg.header.frame_id = PELVIS_FRAME
        msg.pose.position.x    = float(pos[0])
        msg.pose.position.y    = float(pos[1])
        msg.pose.position.z    = float(pos[2])
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])
        return msg

    def camera_info_callback(self, msg: CameraInfo):
        self.fx = msg.k[0]
        self.fy = msg.k[4]
        self.cx = msg.k[2]
        self.cy = msg.k[5]
        self.get_logger().info(
            f'Intrinsics updated: fx={self.fx:.2f} fy={self.fy:.2f} '
            f'cx={self.cx:.2f} cy={self.cy:.2f} ({msg.width}x{msg.height})',
            throttle_duration_sec=10.0
        )

    def pose_callback(self, msg: Detection3DArray):
        if msg.detections:
            det                  = msg.detections[0]
            self.latest_pose     = det.results[0].pose.pose
            self.latest_box_size = det.bbox.size
            self.latest_stamp    = msg.header.stamp
            p = self.latest_pose.position
            s = self.latest_box_size
            self.get_logger().info(
                f'[TRACKING] pos=({p.x:.3f},{p.y:.3f},{p.z:.3f})m  '
                f'box=({s.x:.3f}×{s.y:.3f}×{s.z:.3f})m',
                throttle_duration_sec=1.0,
            )

    def rgb_callback(self, msg: Image):
        now = time.time()
        if now - self.last_publish_time < self.MIN_INTERVAL:
            return
        self.last_publish_time = now

        img = np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3).copy()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        if self.fx is None:
            cv2.putText(img_bgr, 'Waiting for /camera_info...',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            self._publish_image(img_bgr, msg)
            return

        if self.latest_pose is not None and self.latest_box_size is not None:
            img_bgr, g1_cam, g2_cam, R_box_cam = draw_overlay(
                img_bgr, self.latest_pose, self.latest_box_size,
                self.fx, self.fy, self.cx, self.cy)

            T_pelvis_cam = self._get_T_pelvis_cam(self.latest_stamp)
            if T_pelvis_cam is not None:

                # ── Grasp positions in pelvis frame ───────────────────────────
                g1_pelvis = cam_to_pelvis(g1_cam, T_pelvis_cam)
                g2_pelvis = cam_to_pelvis(g2_cam, T_pelvis_cam)

                # Nudge both grasp points: 5cm closer to camera/robot, 5cm lower
                g1_pelvis = g1_pelvis + GRASP_PELVIS_OFFSET
                g2_pelvis = g2_pelvis + GRASP_PELVIS_OFFSET

                # ── Grasp orientations aligned with box face normal ────────────
                # hand Y axis is aligned with ±box_Y_pelvis
                # hand X axis (fingers) kept as close to +X pelvis as possible
                q_left, q_right = compute_grasp_quaternions(R_box_cam, T_pelvis_cam)

                # ── Assign left/right by pelvis Y (higher Y = left) ───────────
                if g1_pelvis[1] >= g2_pelvis[1]:
                    left_pos,  left_quat  = g1_pelvis, q_left
                    right_pos, right_quat = g2_pelvis, q_right
                else:
                    left_pos,  left_quat  = g2_pelvis, q_right
                    right_pos, right_quat = g1_pelvis, q_left

                # ── Diagnostic: box Y axis in pelvis frame ────────────────────
                R_pelvis_cam = T_pelvis_cam[:3, :3]
                box_y_pelvis = R_pelvis_cam @ R_box_cam[:, 1]
                box_y_pelvis = box_y_pelvis / np.linalg.norm(box_y_pelvis)

                # Verify hand Y axes
                R_left_check  = quat_to_rot(left_quat)
                R_right_check = quat_to_rot(right_quat)

                stamp = msg.header.stamp
                self.pub_left.publish(
                    self._make_pose_stamped(left_pos,  left_quat,  stamp))
                self.pub_right.publish(
                    self._make_pose_stamped(right_pos, right_quat, stamp))

                self.get_logger().info(
                    f'\n'
                    f'  [BOX Y in pelvis]: ({box_y_pelvis[0]:.3f},{box_y_pelvis[1]:.3f},{box_y_pelvis[2]:.3f})\n'
                    f'  [GRASP POSITIONS]\n'
                    f'    LEFT  pos  : ({left_pos[0]:.3f},{left_pos[1]:.3f},{left_pos[2]:.3f}) m\n'
                    f'    RIGHT pos  : ({right_pos[0]:.3f},{right_pos[1]:.3f},{right_pos[2]:.3f}) m\n'
                    f'  [ORIENTATION CHECK — hand Y axis in pelvis]\n'
                    f'    left  Y: ({R_left_check[0,1]:.3f},{R_left_check[1,1]:.3f},{R_left_check[2,1]:.3f})'
                    f' ← should be near -{box_y_pelvis.round(2)}\n'
                    f'    right Y: ({R_right_check[0,1]:.3f},{R_right_check[1,1]:.3f},{R_right_check[2,1]:.3f})'
                    f' ← should be near +{box_y_pelvis.round(2)}\n'
                    f'  [QUATERNIONS (x,y,z,w)]\n'
                    f'    LEFT  quat : ({left_quat[0]:.3f},{left_quat[1]:.3f},{left_quat[2]:.3f},{left_quat[3]:.3f})\n'
                    f'    RIGHT quat : ({right_quat[0]:.3f},{right_quat[1]:.3f},{right_quat[2]:.3f},{right_quat[3]:.3f})'
                )

            else:
                self.get_logger().warn(
                    'Skipping grasp publish — TF not yet available.',
                    throttle_duration_sec=2.0,
                )
        else:
            cv2.putText(img_bgr, 'Waiting for tracking pose...',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)

        self._publish_image(img_bgr, msg)

    def _publish_image(self, img_bgr, msg):
        img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        out          = Image()
        out.header   = msg.header
        out.height   = msg.height
        out.width    = msg.width
        out.encoding = 'rgb8'
        out.step     = msg.width * 3
        out.data     = img_rgb.tobytes()
        self.pub_vis.publish(out)


def main():
    rclpy.init()
    node = PoseOverlayVisualizerTracking()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()