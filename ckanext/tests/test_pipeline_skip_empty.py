import pandas as pd
from unittest.mock import MagicMock

from ckanext.portal_metrics.pipeline import MetricsPipeline


def test_pipeline_skips_empty_frames(monkeypatch):
    """No datastore_upsert call is issued when GA4 returns no rows."""

    # --- build fake GA4 exporter that always returns an empty DF -----------
    fake_exporter = MagicMock()
    fake_exporter.fetch_analytics.return_value = pd.DataFrame()
    fake_exporter.fetch_downloads.return_value = pd.DataFrame()
    fake_exporter.to_dataframe.return_value = pd.DataFrame()

    # --- monkey-patch pipeline components ----------------------------------
    monkeypatch.setattr("ckanext.portal_metrics.pipeline.GA4Exporter", lambda *a, **k: fake_exporter)

    fake_ckan = MagicMock()
    fake_ckan.upsert = MagicMock()
    monkeypatch.setattr("ckanext.portal_metrics.pipeline.CkanClient", lambda *a, **k: fake_ckan)

    # --- run ----------------------------------------------------------------
    pipeline = MetricsPipeline(property_id="123", hostname="example.org",
                               context={}, creds=None)
    pipeline.run()

    # --- verify -------------------------------------------------------------
    fake_ckan.upsert.assert_not_called()
