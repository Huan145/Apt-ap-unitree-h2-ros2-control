from types import SimpleNamespace as NS

import pytest

import ros_stubs
from helpers import string
from targets import load_targets


class FakeFuture:
    def __init__(self):
        self.cbs, self.value = [], None

    def add_done_callback(self, cb):
        self.cbs.append(cb)

    def resolve(self, *ok_reasons):
        self.value = NS(results=[NS(successful=ok, reason=why) for ok, why in ok_reasons])
        for cb in self.cbs:
            cb(self)

    def result(self):
        if self.value is None:
            raise RuntimeError('service call failed')
        return self.value


class FakeClient:
    def __init__(self, name):
        self.name, self.ready, self.calls = name, True, []

    def service_is_ready(self):
        return self.ready

    def call_async(self, req):
        fut = FakeFuture()
        self.calls.append((req, fut))
        return fut


@pytest.fixture
def sw():
    import target_switcher
    ros_stubs.Node.client_factory = FakeClient
    n = target_switcher.TargetSwitcher()
    return n


def params(call):
    return dict(call[0].parameters)


def test_client_points_at_the_foundationpose_node(sw):
    assert sw.cli.name == '/foundationpose_node/set_parameters'


def test_switch_sets_the_targets_parameters_once(sw):
    sw.subs['/active_target'](string('workstation'))
    assert len(sw.cli.calls) == 1
    p = params(sw.cli.calls[0])
    assert p == load_targets()['workstation']['foundationpose']       # exactly what the registry says
    assert p['mesh_file_path'].endswith('workstation_one.obj')
    sw.subs['/active_target'](string('workstation'))            # busy: the 1 Hz republish is ignored
    assert len(sw.cli.calls) == 1
    sw.cli.calls[0][1].resolve((True, ''), (True, ''))
    assert sw.applied == 'workstation' and not sw.busy and sw.log.has('info', 'applied')
    sw.subs['/active_target'](string('workstation'))            # applied: still nothing to do
    assert len(sw.cli.calls) == 1


def test_switch_back_restores_klt_mesh_and_no_symmetry(sw):
    sw.subs['/active_target'](string('workstation'))
    sw.cli.calls[0][1].resolve((True, ''), (True, ''))
    sw.subs['/active_target'](string('klt_box'))
    p = params(sw.cli.calls[1])
    assert p['mesh_file_path'].endswith('KLT_box_metres.obj') and p['symmetry_axes'] == ['']


def test_rejected_parameter_is_retried_on_the_next_message(sw):
    sw.subs['/active_target'](string('workstation'))
    sw.cli.calls[0][1].resolve((True, ''), (False, 'bad axis'))
    assert sw.applied is None and sw.log.has('error', 'bad axis')
    sw.subs['/active_target'](string('workstation'))
    assert len(sw.cli.calls) == 2


def test_failed_call_releases_the_lock(sw):
    sw.subs['/active_target'](string('workstation'))
    sw.cli.calls[0][1].cbs[0](sw.cli.calls[0][1])               # future without a result -> exception path
    assert not sw.busy and sw.applied is None
    sw.subs['/active_target'](string('workstation'))
    assert len(sw.cli.calls) == 2


def test_service_not_ready_and_unknown_target(sw):
    sw.cli.ready = False
    sw.subs['/active_target'](string('workstation'))
    assert sw.cli.calls == [] and not sw.busy and sw.log.has('warn', 'not available')
    sw.cli.ready = True
    sw.subs['/active_target'](string('toaster'))
    assert sw.cli.calls == [] and sw.log.has('error', 'toaster')
    sw.subs['/active_target'](string('workstation'))            # service came up: works
    assert len(sw.cli.calls) == 1
