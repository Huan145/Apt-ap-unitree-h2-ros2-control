"""Minimal ROS 2 stand-ins so the perception nodes can be unit-tested without a ROS install.

Messages are auto-vivifying namespaces (msg.pose.position.x = 1.0 just works); nodes record their
subscriptions/publishers, so tests call callbacks directly and read what was published.
"""
import sys
import types
from types import SimpleNamespace


class Msg(SimpleNamespace):
    def __getattr__(self, k):
        if k.startswith('__'):
            raise AttributeError(k)
        v = Msg()
        setattr(self, k, v)
        return v


class Logger:
    def __init__(self):
        self.records = []

    def _log(self, level, msg, **kw):
        self.records.append((level, msg))

    def info(self, m, **kw):  self._log('info', m)
    def warn(self, m, **kw):  self._log('warn', m)
    def error(self, m, **kw): self._log('error', m)
    def debug(self, m, **kw): self._log('debug', m)

    def has(self, level, text):
        return any(l == level and text in m for l, m in self.records)


class Publisher:
    def __init__(self):
        self.msgs = []

    def publish(self, m):
        self.msgs.append(m)


class Clock:
    """Sim clock a test advances with `node.get_clock().t = seconds`."""
    t = 0.0

    def now(self):
        t = self.t
        return SimpleNamespace(nanoseconds=int(round(t * 1e9)),
                               to_msg=lambda: Msg(sec=int(t), nanosec=int(round((t % 1) * 1e9))))


class Node:
    overrides = {}          # parameter overrides, set by a test before it builds a node
    client_factory = None   # fake service client, set by the switcher tests

    def __init__(self, name, **kw):
        self.name, self._params, self.log, self.clock = name, {}, Logger(), Clock()
        self.subs, self.pubs, self.srvs, self.timers = {}, {}, {}, []

    def get_logger(self):                         return self.log
    def declare_parameter(self, name, default=None):
        self._params[name] = Node.overrides.get(name, default)
    def get_parameter(self, name):                return SimpleNamespace(value=self._params[name])
    def create_subscription(self, t, topic, cb, qos): self.subs[topic] = cb
    def create_client(self, t, name):             return Node.client_factory(name)
    def destroy_node(self):                       pass
    def get_clock(self):                          return self.clock
    def set_parameters(self, params):             pass
    def create_service(self, t, name, cb):        self.srvs[name] = cb
    def create_timer(self, period, cb):           self.timers.append((period, cb))

    def create_publisher(self, t, topic, qos):
        self.pubs[topic] = Publisher()
        return self.pubs[topic]


class Parameter:
    Type = SimpleNamespace(BOOL='bool')

    def __init__(self, name, type_=None, value=None):
        self.name, self.value = name, value

    def to_parameter_msg(self):
        return (self.name, self.value)


class Buffer:
    T = None     # TransformStamped-like returned by lookup_transform; None -> LookupException

    def lookup_transform(self, target, source, time, timeout=None):
        if Buffer.T is None:
            raise LookupException('no transform')
        return Buffer.T

    def all_frames_as_string(self):
        return ''


class TransformException(Exception): pass
class LookupException(TransformException): pass
class ConnectivityException(TransformException): pass
class ExtrapolationException(TransformException): pass


def _module(name, **attrs):
    m = types.ModuleType(name)
    m.__dict__.update(attrs)
    sys.modules[name] = m
    return m


_MSG_TYPES = None


def install():
    """Idempotent: registers the stub modules once and returns the message classes."""
    global _MSG_TYPES
    if _MSG_TYPES:
        return _MSG_TYPES
    msg_types = {c: type(c, (Msg,), {}) for c in
                 ('Image', 'CameraInfo', 'Detection3DArray', 'PoseStamped', 'String',
                  'Twist', 'PointStamped', 'Float64', 'JointState')}
    time_m = _module('rclpy.time', Time=type('Time', (), {'from_msg': staticmethod(lambda s: s)}))
    dur_m = _module('rclpy.duration', Duration=lambda seconds=0: seconds)
    _module('rclpy', init=lambda *a, **k: None, shutdown=lambda: None, spin=lambda n: None,
            time=time_m, duration=dur_m)
    _module('rclpy.node', Node=Node)
    _module('rclpy.parameter', Parameter=Parameter)
    _module('rcl_interfaces')
    _module('rcl_interfaces.srv', SetParameters=type('SetParameters', (), {'Request': Msg}))
    _module('sensor_msgs')
    _module('sensor_msgs.msg', Image=msg_types['Image'], CameraInfo=msg_types['CameraInfo'],
            JointState=msg_types['JointState'])
    _module('std_msgs')
    _module('std_msgs.msg', String=msg_types['String'], Float64=msg_types['Float64'])
    _module('vision_msgs')
    _module('vision_msgs.msg', Detection3DArray=msg_types['Detection3DArray'])
    _module('geometry_msgs')
    _module('geometry_msgs.msg', PoseStamped=msg_types['PoseStamped'], Twist=msg_types['Twist'],
            PointStamped=msg_types['PointStamped'])
    _module('std_srvs')
    _module('std_srvs.srv', Trigger=type('Trigger', (), {}))
    _module('rclpy.qos', QoSProfile=lambda **kw: kw, DurabilityPolicy=SimpleNamespace(TRANSIENT_LOCAL='tl'))
    _module('tf2_ros', Buffer=Buffer, TransformListener=lambda buf, node: None,
            LookupException=LookupException, ConnectivityException=ConnectivityException,
            ExtrapolationException=ExtrapolationException, TransformException=TransformException)
    _MSG_TYPES = msg_types
    return msg_types
