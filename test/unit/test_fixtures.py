# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.

from __future__ import annotations

from typing import Any, cast
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
    def _run(os_name: str, worker_cls: MagicMock) -> MagicMock:
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
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        worker_cls = MagicMock()
        self._run("MACOS", worker_cls)

        worker_cls.assert_called_once()
        kwargs = worker_cls.call_args.kwargs
        assert set(kwargs) == {"configuration", "deadline_client"}
        for ec2_only in ("subnet_id", "security_group_id", "instance_profile_name", "ec2_client"):
            assert ec2_only not in kwargs

    def test_starts_and_stops_the_worker(self, monkeypatch) -> None:
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        worker_cls = MagicMock()
        worker = self._run("MACOS", worker_cls)

        worker.start.assert_called_once()
        worker.stop.assert_called_once()

    def test_linux_still_requires_a_subnet(self, monkeypatch) -> None:
        # Guards the branch order: putting the macOS check after the EC2 one, or making it too
        # broad, would silently drop this requirement for AL2023.
        monkeypatch.delenv("SUBNET_ID", raising=False)
        monkeypatch.delenv("USE_DOCKER_WORKER", raising=False)
        monkeypatch.setattr(fixtures.boto3, "client", MagicMock())

        with pytest.raises(AssertionError, match="SUBNET_ID"):
            self._run("AL2023", MagicMock())
