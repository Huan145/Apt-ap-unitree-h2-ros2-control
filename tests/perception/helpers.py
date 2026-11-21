"""Message builders shared by the perception tests (camera: x right, y down, z forward)."""
import numpy as np
from scipy.spatial.transform import Rotation as Rot

import ros_stubs
from ros_stubs import Msg

R_PELVIS_CAM = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]], float)   # camera axes expressed in pelvis
T_PELVIS_CAM = np.array([0.1, 0.0, 0.6])                              # camera origin in pelvis
MSG = ros_stubs.install()


def stamp(sec=12, nanosec=500):
    return Msg(sec=sec, nanosec=nanosec)


def tf_stamped(R=R_PELVIS_CAM, t=T_PELVIS_CAM):
    m = Msg()
    m.transform.translation.x, m.transform.translation.y, m.transform.translation.z = map(float, t)
    (m.transform.rotation.x, m.transform.rotation.y,
     m.transform.rotation.z, m.transform.rotation.w) = Rot.from_matrix(R).as_quat()
    return m


def obj_in_cam(R_obj_pelvis, p_obj_pelvis):
    """(R, t) of an object given in the pelvis frame, as FoundationPose reports it (camera frame)."""
    Rc = R_PELVIS_CAM.T
    return Rc @ R_obj_pelvis, Rc @ (np.asarray(p_obj_pelvis) - T_PELVIS_CAM)


def det_msg(R_obj_cam, t_cam, size, st=None):
    m = MSG['Detection3DArray']()
    m.header.stamp = st or stamp()
    d = Msg()
    d.results = [Msg()]
    p = d.results[0].pose.pose
    p.position.x, p.position.y, p.position.z = map(float, t_cam)
    (p.orientation.x, p.orientation.y,
     p.orientation.z, p.orientation.w) = Rot.from_matrix(R_obj_cam).as_quat()
    d.bbox.size.x, d.bbox.size.y, d.bbox.size.z = size
    m.detections = [d]
    return m


def rgb_msg(w=720, h=360, st=None):
    m = MSG['Image']()
    m.header.stamp, m.header.frame_id = st or stamp(12, 700), 'Camera'
    m.height, m.width, m.encoding = h, w, 'rgb8'
    m.data = np.zeros(h * w * 3, np.uint8).tobytes()
    return m


def cam_info():
    c = MSG['CameraInfo']()
    c.k = [412.31, 0, 360, 0, 412.31, 180, 0, 0, 1]
    c.width, c.height = 720, 360
    return c


def string(data):
    return MSG['String'](data=data)


def xyz(p):
    return np.array([p.x, p.y, p.z])
