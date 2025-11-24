# -*- coding: utf-8 -*-
"""Unit-tests for *ckanext.portal_metrics.cli.metrics*."""
from __future__ import annotations

from types import ModuleType
from typing import Any
import sys

import pytest
from click.testing import CliRunner


# --------------------------------------------------------------------------- #
# Fixtures                                                                    #
# --------------------------------------------------------------------------- #
@pytest.fixture(name="stub_toolkit")
def _fixture_stub_toolkit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub *ckan.plugins.toolkit* so the CLI can resolve `get_action`."""
    tk_mod = ModuleType("ckan.plugins.toolkit")

    def _get_action(name: str):  # noqa: D401
        if name != "get_site_user":
            raise KeyError(name)

        def _impl(_context: dict[str, Any], _data: dict[str, Any]):  # noqa: D401
            return {"name": "stub_sysadmin"}

        return _impl

    tk_mod.get_action = _get_action

    # Make the stub importable under the fully-qualified name
    sys.modules["ckan.plugins.toolkit"] = tk_mod

    # Also patch the already-imported CLI module (if any)
    import ckanext.portal_metrics.cli.metrics as metrics_mod

    monkeypatch.setattr(metrics_mod, "toolkit", tk_mod, raising=True)


@pytest.fixture(name="stub_plugin")
def _fixture_stub_plugin(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace *MetricsCliPlugin* with a tiny object exposing needed attrs."""
    import ckanext.portal_metrics.cli.metrics as metrics_mod

    class _StubPlugin:  # pylint: disable=too-few-public-methods
        ga4_credentials = object()
        hostname = "example.org"
        property_id = "GA-123"

    monkeypatch.setattr(metrics_mod, "MetricsCliPlugin", _StubPlugin, raising=True)


def _patch_pipeline(monkeypatch: pytest.MonkeyPatch, *, raise_exc: bool = False):
    """Inject stub for *MetricsPipeline* and capture its constructor args."""
    import ckanext.portal_metrics.cli.metrics as metrics_mod

    called: dict[str, Any] = {}

    class _StubPipeline:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):  # noqa: D401, ANN001
            called["args"] = args
            called["kwargs"] = kwargs

        def run(self):  # noqa: D401
            if raise_exc:
                raise RuntimeError("boom")

    monkeypatch.setattr(metrics_mod, "MetricsPipeline", _StubPipeline, raising=True)
    return called


# --------------------------------------------------------------------------- #
# Tests                                                                       #
# --------------------------------------------------------------------------- #
def test_fetch_metrics_success(stub_toolkit, stub_plugin, monkeypatch):
    """Pipeline completes → CLI exits with 0 and passes correct parameters."""
    import ckanext.portal_metrics.cli.metrics as metrics_mod

    called = _patch_pipeline(monkeypatch)

    runner = CliRunner()
    result = runner.invoke(metrics_mod.metrics, ["fetch", "--lookback-days", "5"])

    assert result.exit_code == 0, result.output

    # verify constructor signature
    prop_id, host, context, creds, lookback, org = called["args"]
    assert prop_id == "GA-123"
    assert host == "example.org"
    assert context["user"] == "stub_sysadmin"
    assert creds is not None
    assert lookback == 5
    assert org == "portal-metrics"


def test_fetch_metrics_failure(stub_toolkit, stub_plugin, monkeypatch):
    """Pipeline raises → CLI exits with code 1 and prints the error."""
    import ckanext.portal_metrics.cli.metrics as metrics_mod

    _patch_pipeline(monkeypatch, raise_exc=True)

    runner = CliRunner()
    result = runner.invoke(metrics_mod.metrics, ["fetch"])  # default catch_exceptions=True

    # Click converted the ClickException to SystemExit(1)
    assert result.exit_code == 1
    assert isinstance(result.exception, SystemExit)
    # the original RuntimeError message must be shown to the user
    assert "boom" in result.output
