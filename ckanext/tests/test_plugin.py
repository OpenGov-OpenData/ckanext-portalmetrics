# -*- coding: utf-8 -*-
"""Unit-tests for *ckanext.portal_metrics.plugin.MetricsCliPlugin*.

The tests focus on the `configure` method because this is the only logic
inside the plug-in (registration is exercised indirectly).  All calls to
Google OAuth are patched so the test-suite does **not** require a real
service-account key.
"""
from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.fixture(name="plugin")
def _fixture_plugin():
    """Return a *fresh* MetricsCliPlugin instance for every test."""
    from ckanext.portal_metrics.plugin import MetricsCliPlugin

    # ------------------------------------------------------------------
    # 1)  Forget the previously created singleton
    # ------------------------------------------------------------------
    if hasattr(MetricsCliPlugin, "_instance"):
        MetricsCliPlugin._instance = None

    # ------------------------------------------------------------------
    # 2)  Create a brand-new instance
    # ------------------------------------------------------------------
    instance = MetricsCliPlugin()

    # ------------------------------------------------------------------
    # 3)  Reset per-test state on that instance
    # ------------------------------------------------------------------
    instance.ga4_credentials = None
    instance.property_id = None
    instance.hostname = None

    return instance


@pytest.fixture(autouse=True)
def _patch_google(monkeypatch: pytest.MonkeyPatch):
    """Stub *google.oauth2.service_account.Credentials.from_service_account_info*."""
    # Create a tiny dummy object that will stand in for the credentials
    _dummy_credentials = SimpleNamespace(token="dummy")

    class _FakeCredentials:  # pylint: disable=too-few-public-methods
        @staticmethod
        def from_service_account_info(_info: dict[str, Any]):  # noqa: D401, ANN001
            return _dummy_credentials

    monkeypatch.setattr(
        "ckanext.portal_metrics.plugin.service_account.Credentials",
        _FakeCredentials,
        raising=True,
    )
    return _dummy_credentials


# --------------------------------------------------------------------------- #
#                                TESTS                                        #
# --------------------------------------------------------------------------- #
def test_configure_success(plugin, _patch_google):
    """Valid config → credentials are stored and host / property parsed."""
    ini = {
        "ckanext.portal_metrics.ga4_property_id": "GA-42",
        "ckanext.portal_metrics.ga4_client_email": "svc@example.com",
        "ckanext.portal_metrics.ga4_private_key": "-----BEGIN PRIVATE KEY-----\nABC\n-----END PRIVATE KEY-----\n",
        "ckan.site_url": "https://data.houstontx.gov",
    }

    plugin.configure(ini)

    assert plugin.property_id == "GA-42"
    assert plugin.hostname == "data.houstontx.gov"
    # type: ignore[arg-type] – fixture returns the dummy credentials object
    assert plugin.ga4_credentials is _patch_google  # noqa: E721


def test_configure_missing_values(plugin, caplog):
    """Missing mandatory settings → credentials remain *None* and error is logged."""
    caplog.set_level(logging.ERROR)

    ini = {
        # property_id intentionally missing
        "ckanext.portal_metrics.ga4_client_email": "svc@example.com",
        "ckanext.portal_metrics.ga4_private_key": "k",
        "ckan.site_url": "https://data.houstontx.gov",
    }

    plugin.configure(ini)

    assert plugin.ga4_credentials is None
    assert "Missing GA4 service account config values" in caplog.text


def test_configure_bad_key(plugin, monkeypatch, caplog):
    """Google helper raises → plugin keeps running but credentials == None."""
    caplog.set_level(logging.ERROR)

    # Force the patched Google helper to raise
    def _raise(_info):  # noqa: ANN001
        raise ValueError("malformed key")

    monkeypatch.setattr(
        "ckanext.portal_metrics.plugin.service_account.Credentials.from_service_account_info",
        _raise,
        raising=True,
    )

    ini = {
        "ckanext.portal_metrics.ga4_property_id": "GA-42",
        "ckanext.portal_metrics.ga4_client_email": "svc@example.com",
        "ckanext.portal_metrics.ga4_private_key": "k",
        "ckan.site_url": "https://data.houstontx.gov",
    }

    plugin.configure(ini)

    assert plugin.ga4_credentials is None
    assert "Failed to create GA4 credentials" in caplog.text
