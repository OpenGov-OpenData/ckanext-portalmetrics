from datetime import date
from types import SimpleNamespace
from unittest.mock import patch
import pandas as pd
import pytest
import ckanext.portal_metrics.pipeline as pl


@pytest.fixture(scope="session")
def tk_stub():
    """Minimal replacement for ckan.plugins.toolkit."""
    return SimpleNamespace(ObjectNotFound=type("ObjNF", (Exception,), {}))


@pytest.fixture(autouse=True)
def _use_stub_toolkit(monkeypatch, tk_stub):
    """Patch pipeline.tk with the stub for every test."""
    monkeypatch.setattr(pl, "tk", tk_stub, raising=False)


@pytest.fixture(scope="session")
def ga4_resp():
    def make_hdr(n):
        return SimpleNamespace(name=n)

    def make_val(v):
        return SimpleNamespace(value=v)

    def row(dims, mets):
        return SimpleNamespace(
            dimension_values=[make_val(x) for x in dims],
            metric_values=[make_val(x) for x in mets],
        )

    return SimpleNamespace(
        dimension_headers=[make_hdr("date"), make_hdr("pagePath")],
        metric_headers=[make_hdr("totalUsers")],
        rows=[row(["20240101", "/foo"], [5]), row(["20240102", "/bar"], [10])],
    )


@pytest.fixture
def df_raw():
    return pd.DataFrame(
        {
            "date": ["20240101", "20240102"],
            "pagePath": ["/foo", "/bar"],
            "totalUsers": [5, 10],
            "userEngagementDuration": [100, 200],
            "screenPageViews": [1, 2],
            "averageSessionDuration": [10.0, 20.0],
            "newUsers": [1, 2],
            "activeUsers": [1, 2],
            "pageTitle": ["Foo", "Bar"],
        }
    )


def test_ga4_to_dataframe(ga4_resp):
    exp = pl.GA4Exporter(creds=object(), property_id="pid", hostname="h")
    df = exp.to_dataframe(ga4_resp)
    assert list(df.columns) == ["date", "pagePath", "totalUsers"]
    assert len(df) == 2


@patch.object(pl, "BetaAnalyticsDataClient")
def test_ga4_pagination(mock_cli, ga4_resp):
    # First page has rows, second is empty to stop the loop
    empty_resp = SimpleNamespace(
        dimension_headers=ga4_resp.dimension_headers,
        metric_headers=ga4_resp.metric_headers,
        rows=[],
    )
    mock_cli.return_value.run_report.side_effect = [ga4_resp, empty_resp]
    exp = pl.GA4Exporter(creds=object(), property_id="pid", hostname="h")
    exp.client = mock_cli.return_value
    df = exp.fetch_analytics("2024-01-01", "2024-01-02")
    assert len(df) == 2


@patch.object(pl, "BetaAnalyticsDataClient")
def test_ga4_retry(mock_cli, ga4_resp, monkeypatch):
    empty_resp = SimpleNamespace(
        dimension_headers=ga4_resp.dimension_headers,
        metric_headers=ga4_resp.metric_headers,
        rows=[],
    )
    mock_cli.return_value.run_report.side_effect = [
        RuntimeError("boom"),
        ga4_resp,
        empty_resp,
    ]
    monkeypatch.setattr(pl.time, "sleep", lambda *_: None)
    exp = pl.GA4Exporter(creds=object(), property_id="pid", hostname="h")
    exp.client = mock_cli.return_value
    df = exp.fetch_analytics("2024-01-01", "2024-01-02")
    assert len(df) == 2


def test_page_size_passthrough(monkeypatch, ga4_resp):
    exporter = pl.GA4Exporter(creds=object(), property_id="pid", hostname="h")
    # Build an empty response that keeps the required header attributes
    empty_resp = SimpleNamespace(
        dimension_headers=ga4_resp.dimension_headers,
        metric_headers=ga4_resp.metric_headers,
        rows=[],
    )

    # Force _fetch_all to finish immediately
    monkeypatch.setattr(exporter, "_fetch_page", lambda *a, **k: empty_resp)

    # Should accept provided page_size and exit without error
    exporter._fetch_all("2020-01-01", "2020-01-02", exporter._fetch_analytics, page_size=99)


def test_metrics_processor(df_raw):
    proc = pl.MetricsProcessor()
    cleaned = proc.clean(df_raw)
    # Use a copy of the map to avoid mutation by pop("group_by")
    result = proc.aggregate(cleaned, dict(proc.ANALYTICS_AGG_MAP))
    assert "page_path" in cleaned
    assert not result.empty


class _Sentinel(Exception):
    """Local stand-in for tk.ObjectNotFound during parametrisation."""


@pytest.mark.parametrize(
    "package_show_result, expect_create",
    [({"id": "ds"}, False), (_Sentinel, True)],
)
def test_ensure_dataset(package_show_result, expect_create, tk_stub):
    tk_stub.ObjectNotFound = _Sentinel
    calls = {"create": 0}

    def pkg_show(*_):
        if package_show_result is _Sentinel:
            raise _Sentinel()
        return package_show_result

    def pkg_create(*_):
        calls["create"] += 1
        return {"id": "ds"}

    tk_stub.get_action = lambda n: {
        "package_show": pkg_show,
        "package_create": pkg_create,
    }[n]

    cid = pl.CkanClient({"user": "u"})
    assert cid.ensure_dataset("org") == "ds"
    assert bool(calls["create"]) is expect_create


def test_find_resource(tk_stub):
    tk_stub.get_action = lambda n: lambda *_: {"resources": [{"name": "Portal Analytics Data", "id": "rid"}]}
    assert pl.CkanClient({"user": "u"}).find_resource("ds") == "rid"


def test_upsert_retry(monkeypatch, df_raw, tk_stub):
    attempts = {"n": 0}

    def ds_create(*_):
        return {"resource_id": "rid"}

    def ds_upsert(*_):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise RuntimeError
        return {"success": True}

    tk_stub.get_action = lambda n: {
        "datastore_create": ds_create,
        "datastore_upsert": ds_upsert,
    }[n]
    monkeypatch.setattr(pl.time, "sleep", lambda *_: None)

    pl.CkanClient({"user": "u"}).upsert(
        df_raw, "ds", name="Test", datadict=[], pk=["date"]
    )
    assert attempts["n"] == 2


def test_resource_cache():
    class Dummy:
        def list_packages_with_resources(self, *_, **__):
            return [
                {
                    "organization": {"id": "org", "name": "ORG"},
                    "groups": [{"display_name": "grp"}],
                    "resources": [{"id": "rid", "format": "CSV", "name": "foo"}],
                }
            ]

    cache = pl.ResourceCache(Dummy())
    assert cache.get("rid")["format"] == "CSV"


def test_resource_cache_pagination(monkeypatch):
    pages = [
        [
            {"organization": {"id": "x"}, "resources": [{"id": "a"}], "groups": []},
            {"organization": {"id": "x"}, "resources": [{"id": "b"}], "groups": []},
        ],
        [],
    ]

    class Dummy:
        def list_packages_with_resources(self, limit, offset):
            return pages.pop(0)

    cache = pl.ResourceCache(Dummy())
    assert cache.get("a") != {}
    assert cache.get("b") != {}


@patch.object(pl.CkanClient, "ensure_dataset", return_value="ds")
@patch.object(pl.CkanClient, "find_resource", return_value="rid")
@patch.object(pl.CkanClient, "upsert")
@patch.object(pl.CkanClient, "list_packages_with_resources", return_value=[])
@patch.object(pl.ResourceCache, "get", return_value={})
@patch.object(pl.MetricsProcessor, "aggregate", side_effect=lambda df, *_, **__: df)
@patch.object(pl.MetricsProcessor, "clean",
              side_effect=lambda df: df.assign(page_path=df.get("pagePath", "")))
@patch.object(
    pl.GA4Exporter,
    "fetch_downloads",
    return_value=pd.DataFrame(
        {
            "date": [date.today().isoformat()],
            "pagePath": ["/foo"],
            "fileName": ["report /dataset/x/resource/123e4567-e89b-12d3-a456-426614174000.csv"],
        }
    ),
)
@patch.object(
    pl.GA4Exporter,
    "fetch_analytics",
    return_value=pd.DataFrame(
        {"date": [date.today().isoformat()], "pagePath": ["/foo"]}
    ),
)
def test_pipeline(*_mocks):
    pl.MetricsPipeline("pid", "host", {"user": "u"}, object(), 1, "org").run()


def test_retry_exhaust(monkeypatch):
    exporter = pl.GA4Exporter(creds=object(), property_id="pid", hostname="h")
    exporter.MAX_RETRIES = 2

    # Make every _fetch_page call raise
    def _raise(*_a, **_k):
        raise RuntimeError("boom")

    monkeypatch.setattr(exporter, "_fetch_page", _raise)
    monkeypatch.setattr(pl.time, "sleep", lambda **kwargs: None)
    with pytest.raises(RuntimeError):
        exporter.fetch_analytics("2020-01-01", "2020-01-02")


def test_upsert_skip_create(monkeypatch):
    calls = {"upsert": 0}

    def ds_upsert(*_):
        calls["upsert"] += 1
        return {"success": True}

    tk = SimpleNamespace(get_action=lambda n: ds_upsert if n == "datastore_upsert" else None)
    monkeypatch.setattr(pl, "tk", tk, raising=False)
    cli = pl.CkanClient({"user": "u"})
    cli.upsert(
        pd.DataFrame({"x": [1]}),
        dataset_id="ds",
        resource_id="rid",
        name="Test",
        datadict=[],
        pk=["x"],
    )
    assert calls["upsert"] == 1
