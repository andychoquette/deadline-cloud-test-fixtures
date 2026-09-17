# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from __future__ import annotations

import inspect
from typing import Any, ClassVar, cast
from unittest.mock import MagicMock

import pytest

from deadline_test_fixtures import fixtures
from deadline_test_fixtures.deadline.worker import (
    LocalMacWorker,
    PosixInstanceBuildWorker,
    WindowsInstanceBuildWorker,
)
from deadline_test_fixtures.models import OperatingSystem


def _worker_type_for(os_name: str) -> type:
    """Resolve the ec2_worker_type fixture for one operating system.

    The fixture only reads `operating_system` off the request, so a stub request is enough and
    avoids standing up the whole session-scoped fixture graph.
    """
    request = MagicMock()
    request.getfixturevalue.return_value = OperatingSystem(name=cast(Any, os_name))
    return next(cast(Any, fixtures.ec2_worker_type).__wrapped__(request))


class RecordingMacWorker(LocalMacWorker):
    """A real LocalMacWorker subclass that records construction instead of touching the host.

    A MagicMock will not do: the fixture issubclass-checks the type it is handed, which is the
    point of that check.
    """

    instances: ClassVar[list[RecordingMacWorker]] = []

    def __init__(self, **kwargs: Any) -> None:
        type(self).instances.append(self)
        self.kwargs = kwargs
        self.started = 0
        self.stopped = 0

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1

    @classmethod
    def reset(cls) -> type[RecordingMacWorker]:
        cls.instances = []
        return cls


class TestEc2WorkerType:
    @pytest.mark.parametrize(
        ("os_name", "expected"),
        [
            ("AL2023", PosixInstanceBuildWorker),
            ("WIN2022", WindowsInstanceBuildWorker),
            ("MACOS", LocalMacWorker),
        ],
    )
    def test_selects_the_worker_for_the_operating_system(
        self, os_name: str, expected: type
    ) -> None:
        assert _worker_type_for(os_name) is expected

    def test_rejects_an_unknown_operating_system(self) -> None:
        # The message is what a suite author sees, so it must name every option that works --
        # omitting MACOS is what made the macOS branch look unsupported.
        with pytest.raises(ValueError, match="MACOS"):
            _worker_type_for("SOLARIS")

    def test_macos_worker_is_not_an_ec2_worker(self) -> None:
        # LocalMacWorker configures the agent on the test host rather than provisioning an
        # instance, which is why the `worker` fixture must not take the EC2 path for MACOS: there
        # is no subnet, security group or instance profile to supply.
        from deadline_test_fixtures.deadline.worker import EC2InstanceWorker

        assert not issubclass(LocalMacWorker, EC2InstanceWorker)


class TestWorkerFixtureOnMacos:
    """The `worker` fixture must not take the EC2 path for MACOS.

    This is the half that was actually broken: `ec2_worker_type` yielding LocalMacWorker is not
    enough, because the EC2 branch asserts on SUBNET_ID and SECURITY_GROUP_ID, which a host running
    the agent locally has not got.
    """

    @staticmethod
    def _run(os_name: str, worker_cls: Any) -> Any:
        request = MagicMock()
        request.getfixturevalue.side_effect = lambda name: (
            OperatingSystem(name=cast(Any, os_name)) if name == "operating_system" else MagicMock()
        )
        request.session.testsfailed = 0
        gen = cast(Any, fixtures.worker).__wrapped__(request, MagicMock(), worker_cls)
        worker = next(gen)
        next(gen, None)  # drive teardown so stop_worker runs
        return worker

    def test_builds_the_local_worker_without_any_ec2_inputs(self, monkeypatch) -> None:
        # No SUBNET_ID / SECURITY_GROUP_ID in the environment: the EC2 branch would assert.
        monkeypatch.delenv("SUBNET_ID", raising=False)
        monkeypatch.delenv("SECURITY_GROUP_ID", raising=False)
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setenv("USE_LOCAL_MAC_WORKER", "true")
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        worker_cls = RecordingMacWorker.reset()
        self._run("MACOS", worker_cls)

        assert len(worker_cls.instances) == 1
        kwargs = worker_cls.instances[0].kwargs
        assert set(kwargs) == {"configuration", "deadline_client"}
        for ec2_only in ("subnet_id", "security_group_id", "instance_profile_name", "ec2_client"):
            assert ec2_only not in kwargs
        # Bound against the real signature, not just compared to a literal set: worker_cls is a
        # MagicMock, so without this a rename or a new required field on LocalMacWorker would keep
        # this test green while the fixture raised TypeError.
        inspect.signature(LocalMacWorker).bind(**kwargs)

    def test_starts_and_stops_the_worker(self, monkeypatch) -> None:
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setenv("USE_LOCAL_MAC_WORKER", "true")
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        worker_cls = RecordingMacWorker.reset()
        worker = self._run("MACOS", worker_cls)

        assert worker.started == 1
        assert worker.stopped == 1
        inspect.signature(LocalMacWorker).bind(**worker.kwargs)

    def test_linux_still_requires_a_subnet(self, monkeypatch) -> None:
        # Guards the branch order: putting the macOS check after the EC2 one, or making it too
        # broad, would silently drop this requirement for AL2023.
        monkeypatch.delenv("SUBNET_ID", raising=False)
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        # A real EC2 worker class, not a MagicMock: the EC2InstanceWorker check now runs before
        # the SUBNET_ID assert, so a mock would exercise a different failure than the one named.
        with pytest.raises(AssertionError, match="SUBNET_ID"):
            self._run("AL2023", PosixInstanceBuildWorker)

    def test_refuses_macos_without_the_opt_in(self, monkeypatch) -> None:
        # The whole point of the gate: a suite that adds a macos param must not reconfigure the
        # machine running the tests without someone saying so.
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.delenv("USE_LOCAL_MAC_WORKER", raising=False)
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        worker_cls = RecordingMacWorker.reset()
        # RuntimeError, not AssertionError: this module ships as a pytest plugin, so an assert here
        # would vanish under python -O and the gate would fail open on the destructive path.
        with pytest.raises(RuntimeError, match="USE_LOCAL_MAC_WORKER"):
            self._run("MACOS", worker_cls)
        assert worker_cls.instances == []

    def test_rejects_a_non_mac_override_for_macos(self, monkeypatch) -> None:
        # ec2_worker_type is the documented override point, so the mismatch has to be named here
        # rather than surfacing as a missing keyword deep in __init__.
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setenv("USE_LOCAL_MAC_WORKER", "true")
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        with pytest.raises(AssertionError, match="override it with a LocalMacWorker subclass"):
            self._run("MACOS", cast(Any, PosixInstanceBuildWorker))

    def test_rejects_a_mac_override_for_linux(self, monkeypatch) -> None:
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setenv("SUBNET_ID", "subnet-0")
        monkeypatch.setenv("SECURITY_GROUP_ID", "sg-0")
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        with pytest.raises(AssertionError, match="not an EC2InstanceWorker"):
            self._run("AL2023", cast(Any, LocalMacWorker))

    def test_refuses_docker_and_macos_together(self, monkeypatch) -> None:
        # Ordering the branches instead would hand back a Linux container for a macos param and
        # report the macos test ids as passing.
        monkeypatch.setenv("USE_DOCKER_WORKER", "true")
        monkeypatch.setenv("USE_LOCAL_MAC_WORKER", "true")
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        with pytest.raises(RuntimeError, match="not compatible with operating_system MACOS"):
            self._run("MACOS", RecordingMacWorker.reset())

    def test_tolerates_a_non_class_override(self, monkeypatch) -> None:
        # ec2_worker_type never required a class, so a factory that used to work must not now die
        # on `issubclass() arg 1 must be a class` from inside the fixture.
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setenv("USE_LOCAL_MAC_WORKER", "true")
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        built: list = []

        def factory(**kwargs: Any) -> Any:
            built.append(kwargs)
            return MagicMock()

        self._run("MACOS", factory)
        assert len(built) == 1
        assert set(built[0]) == {"configuration", "deadline_client"}


class TestOperatingSystemFixtureGate:
    """The opt-in is enforced where MACOS is first selected, not only where the host is modified.

    `worker` depends on `worker_config`, which depends on `deadline_resources`; pytest resolves all
    of those before `worker`'s body runs. A gate only in `worker` therefore fires after the
    bootstrap stack, farm, queue and fleet already exist -- it stops the install, but not the bill.
    """

    @staticmethod
    def _run(param: str) -> Any:
        request = MagicMock()
        request.param = param
        return cast(Any, fixtures.operating_system).__wrapped__(request)

    def test_refuses_macos_without_the_opt_in(self, monkeypatch) -> None:
        monkeypatch.delenv("USE_LOCAL_MAC_WORKER", raising=False)
        with pytest.raises(RuntimeError, match="USE_LOCAL_MAC_WORKER"):
            self._run("macos")

    def test_allows_macos_with_the_opt_in(self, monkeypatch) -> None:
        monkeypatch.setenv("USE_LOCAL_MAC_WORKER", "true")
        assert self._run("macos").name == "MACOS"

    @pytest.mark.parametrize(
        ("param", "expected"),
        [("linux", "AL2023"), ("windows", "WIN2022")],
    )
    def test_leaves_the_other_platforms_ungated(self, monkeypatch, param, expected) -> None:
        # A gate that fired for every parametrization would break every existing suite, none of
        # which sets this variable.
        monkeypatch.delenv("USE_LOCAL_MAC_WORKER", raising=False)
        assert self._run(param).name == expected


class TestDeadlineResourcesFixtureSignature:
    def test_does_not_depend_on_the_operating_system(self) -> None:
        # `operating_system` reads request.param with no fallback, so it resolves only under
        # indirect parametrization. Declaring it here would break any suite that uses
        # `deadline_resources` on its own -- at setup, with an AttributeError naming a fixture the
        # suite never asked for.
        params = inspect.signature(cast(Any, fixtures.deadline_resources).__wrapped__).parameters
        assert "operating_system" not in params
