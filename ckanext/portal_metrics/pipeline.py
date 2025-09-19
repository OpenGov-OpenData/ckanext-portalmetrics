import re
import time
import logging
from datetime import date, timedelta

import pandas as pd
from google.analytics.data_v1beta import (
    BetaAnalyticsDataClient,
    FilterExpression,
)
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
    Filter,
)
from ckan.plugins import toolkit as tk


# -----------------------------------------------------------------------------
# Logging config
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# GA4 exporter
# -----------------------------------------------------------------------------
class GA4Exporter:
    """
    Google Analytics 4 (GA4) Data Exporter with transparent pagination support.

    This class provides methods to fetch analytics data from the GA4 API,
    handling pagination automatically to ensure all rows are retrieved.
    It supports conversion of API responses into pandas DataFrames for further processing.

    Usage Example:
        exporter = GA4Exporter()
        df = exporter.fetch_all("2025-01-01", "2025-01-31")
        # df now contains all rows for the given date range, regardless of GA4 API page size limits.

    Key Features:
        - Transparent pagination using offset/limit parameters.
        - Configurable page size (default: 10,000, GA4 API maximum).
        - Returns a single concatenated DataFrame with all results.
        - Logs progress for each page fetched.
        - Maintains compatibility with existing code via `to_dataframe`.
    """
    MAX_GA_PAGE_SIZE = 10000
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2  # seconds

    DIMENSIONS = [
        "date", "pagePath", "pageTitle", "hostName"
    ]
    METRICS = [
        "screenPageViews", "totalUsers", "userEngagementDuration",
        "activeUsers", "averageSessionDuration", "newUsers"
    ]

    def __init__(self, creds, property_id, hostname):
        self.property_id = property_id
        self.hostname = hostname
        self.client = BetaAnalyticsDataClient(credentials=creds)

    def _fetch_page(self, start_date, end_date, offset, limit):
        """
        Fetch a single page of results from GA4 with retry logic and error logging.
        """
        req = RunReportRequest(
            property=f"properties/{self.property_id}",
            dimensions=[Dimension(name=n) for n in self.DIMENSIONS],
            metrics=[Metric(name=m) for m in self.METRICS],
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimension_filter=FilterExpression(
                filter=Filter(
                    field_name="hostName",
                    string_filter=Filter.StringFilter(
                        match_type=Filter.StringFilter.MatchType.EXACT,
                        value=self.hostname,
                        case_sensitive=False
                    )
                )
            ),
            offset=offset,
            limit=limit,
        )

        for attempt in range(1, self.MAX_RETRIES + 1):
            try:
                return self.client.run_report(req)
            except Exception as exc:
                logger.error(
                    "GA4 API call failed (attempt %d/%d): %s",
                    attempt, self.MAX_RETRIES, exc
                )
                if attempt == self.MAX_RETRIES:
                    raise
                time.sleep(self.RETRY_BACKOFF * attempt)

    def fetch_all(self, start_date, end_date, page_size=MAX_GA_PAGE_SIZE):
        """
        Fetch all rows from GA4 for the specified date range, transparently handling pagination.

        :param start_date: Start date (YYYY-MM-DD)
        :param end_date: End date (YYYY-MM-DD)
        :param page_size: Number of rows per page (max 10000, default 10000)
        :return: pandas.DataFrame containing all rows for the query

        This method will repeatedly call the GA4 API with increasing offsets until all data is retrieved.
        """
        all_rows = []
        offset = 0
        total_rows = 0

        while True:
            resp = self._fetch_page(start_date, end_date, offset, page_size)
            df = self.to_dataframe(resp)
            if df.empty:
                break
            all_rows.append(df)
            row_count = len(df)
            total_rows += row_count
            logger.info("Fetched %d rows from GA4 (offset=%d)", row_count, offset)
            if row_count < page_size:
                break
            offset += page_size

        if all_rows:
            logger.info("Total rows fetched from GA4: %d", total_rows)
            return pd.concat(all_rows, ignore_index=True)
        else:
            logger.warning("No data returned from GA4 for the given date range.")
            return pd.DataFrame(columns=self.DIMENSIONS + self.METRICS)

    def to_dataframe(self, resp):
        dims = [h.name for h in resp.dimension_headers]
        mets = [h.name for h in resp.metric_headers]
        records = []
        for row in resp.rows:
            rec = {}
            for name, val in zip(dims, row.dimension_values):
                rec[name] = val.value
            for name, val in zip(mets, row.metric_values):
                rec[name] = val.value
            records.append(rec)
        return pd.DataFrame(records)


# -----------------------------------------------------------------------------
# Data cleaner + aggregator
# -----------------------------------------------------------------------------
class MetricsProcessor:
    INT_COLS = [
        "userEngagementDuration", "totalUsers",
        "screenPageViews", "averageSessionDuration",
        "newUsers", "activeUsers"
    ]
    RE_VIEW = re.compile(r"/view/.*$")
    RE_RESOURCE = re.compile(r"((?:/resource/[a-f0-9-]+))/.*")

    AGG_MAP = {
        "user_engagement_duration": ("userEngagementDuration", "sum"),
        "total_users": ("totalUsers", "sum"),
        "new_users": ("newUsers", "sum"),
        "screen_page_views": ("screenPageViews", "sum"),
        "active_users": ("activeUsers", "sum"),
        "avg_session_duration": ("averageSessionDuration", "mean"),
        "page_title": ("pageTitle", lambda s: " | ".join(sorted({x for x in s if pd.notna(x) and x}))),
    }

    def clean(self, df):
        # drop unused
        df = df.drop(columns=["hostName"], errors="ignore")
        # parse date
        df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce").dt.date
        # ints
        ints = set(self.INT_COLS) & set(df.columns)
        df[list(ints)] = (df[list(ints)]
                          .apply(pd.to_numeric, errors="coerce", downcast='integer')
                          .fillna(0))

        # page_path
        df["page_path"] = (
            df.get("pagePath", "")
            .fillna("")
            .str.replace(self.RE_VIEW, "", regex=True)
            .str.replace(self.RE_RESOURCE, r"\1", regex=True)
            .str.strip()
        )
        return df

    def aggregate(self, df):
        # only keep aggregations for available columns
        agg = {k: v for k, v in self.AGG_MAP.items() if v[0] in df.columns}
        by = [c for c in ("date", "page_path") if c in df.columns]
        return df.groupby(by, dropna=False).agg(**agg).reset_index()


# -----------------------------------------------------------------------------
# CKAN cache + client
# -----------------------------------------------------------------------------
class ResourceCache:
    """
    Loads CKAN’s current_package_list_with_resources (paginated)
    via the provided CkanClient instance, then caches resource metadata.
    """

    def __init__(self, ckan_client):
        """
        :param ckan_client: an instance of CkanClient
        """
        self.ckan_client = ckan_client
        self._cache = {}
        self._warm()

    def _warm(self):
        offset = 0
        page_size = 2000

        while True:
            packages = self.ckan_client.list_packages_with_resources(
                limit=page_size, offset=offset
            )
            if not packages:
                break

            for pkg in packages:
                org = pkg.get("organization") or {}
                org_id = org.get("id", "")
                org_name = org.get("name", "")

                for res in pkg.get("resources", []):
                    self._cache[res["id"]] = {
                        "org_id": org_id,
                        "org_name": org_name,
                        "format": res.get("format", ""),
                        "page_title": res.get("name", ""),
                    }

            # next page
            if len(packages) < page_size:
                break
            offset += page_size

        logger.info("Resource cache loaded: %d items", len(self._cache))

    def get(self, resource_id):
        """Return metadata dict or empty if not found."""
        return self._cache.get(resource_id, {})


class CkanClient:
    MAX_RETRIES = 3
    RETRY_BACKOFF = 2
    MAX_CKAN_UPSERT_BATCH_SIZE = 10000
    DATA_DICT = [
        {"id": "date", "type": "date"},
        {"id": "page_path", "type": "text"},
        {"id": "user_engagement_duration", "type": "int"},
        {"id": "total_users", "type": "int"},
        {"id": "new_users", "type": "int"},
        {"id": "screen_page_views", "type": "int"},
        {"id": "active_users", "type": "int"},
        {"id": "avg_session_duration", "type": "float"},
        {"id": "page_title", "type": "text"},
        {"id": "org_id", "type": "text"},
        {"id": "org_name", "type": "text"},
        {"id": "format", "type": "text"},
    ]

    def __init__(self, context):
        self.context = context

    def ensure_dataset(self, owner_org):
        data = {"id": "portal-metrics"}
        try:
            result = tk.get_action('package_show')(self.context, data)
            return result["id"]
        except tk.ObjectNotFound:
            meta = {
                "owner_org": owner_org,
                "name": "portal-metrics",
                "private": False,
                "title": "Portal Metrics",
                "notes": "Contains portal analytics."
            }
            result = tk.get_action('package_create')(self.context, meta)
            return result["id"]

    def find_resource(self, dataset_id, name="Portal Analytics Data"):
        data = {"id": dataset_id}
        result = tk.get_action('package_show')(self.context, data)
        for r in result.get("resources", []):
            if r["name"] == name:
                return r["id"]

    def upsert(self, df, dataset_id, resource_id=None):
        records = df.to_dict(orient="records")
        if not resource_id:
            payload = {
                "resource": {
                    "package_id": dataset_id,
                    "name": "Portal Analytics Data",
                    "format": "csv"
                },
                "fields": self.DATA_DICT,
                "primary_key": ["date", "page_path"],
                "records": records[:25],
                "force": True
            }
            result = tk.get_action('datastore_create')(self.context, payload)
            resource_id = result["resource_id"]

        for i in range(0, len(records), self.MAX_CKAN_UPSERT_BATCH_SIZE):
            batch = records[i:i + self.MAX_CKAN_UPSERT_BATCH_SIZE]
            logger.info("Upserting rows %d–%d", i, i + len(batch))
            payload = {
                "resource_id": resource_id,
                "records": batch,
                "method": "upsert"
            }
            for attempt in range(1, self.MAX_RETRIES + 1):
                try:
                    tk.get_action('datastore_upsert')(self.context, payload)
                    break  # Success, move to next batch
                except Exception as exc:
                    logger.error(
                        "CKAN datastore_upsert failed (attempt %d/%d): %s",
                        attempt, self.MAX_RETRIES, exc
                    )
                    if attempt == self.MAX_RETRIES:
                        logger.error("Giving up on this batch after %d attempts.", self.MAX_RETRIES)
                        raise
                    time.sleep(self.RETRY_BACKOFF * attempt)
        logger.info("Upsert complete (%d rows)", len(records))

    def list_packages_with_resources(self, limit=1000, offset=0):
        # There is no direct action for this, so fallback to package_search
        data = {
            "rows": limit,
            "start": offset,
            "include_private": True
        }
        result = tk.get_action('package_search')(self.context, data)
        return result.get("results", [])


# -----------------------------------------------------------------------------
# Pipeline orchestrator
# -----------------------------------------------------------------------------
class MetricsPipeline:
    def __init__(self, property_id, hostname, context, creds, lookback_days=3, owner_org=""):
        self.lookback = lookback_days
        self.owner_org = owner_org
        self.ga4 = GA4Exporter(property_id=property_id, hostname=hostname, creds=creds)
        self.proc = MetricsProcessor()
        self.ckan = CkanClient(context)
        self.cache = ResourceCache(self.ckan)

    def run(self):
        start_date = (date.today() - timedelta(days=self.lookback)).isoformat()
        end_date = date.today().isoformat()
        # 1) Extract & transform
        df = self.ga4.fetch_all(start_date, end_date)
        df = self.proc.clean(df)
        df = self.proc.aggregate(df)

        # 2) Enrich titles + org/format
        df["uuid"] = df["page_path"].str.extract(r"/resource/([0-9a-fA-F\-]{36})")[0]
        meta = df["uuid"].map(self.cache.get).apply(pd.Series)
        df = pd.concat([df, meta], axis=1).drop(columns=["uuid"])
        # 3) Upsert to CKAN
        ds_id = self.ckan.ensure_dataset(self.owner_org)
        res_id = self.ckan.find_resource(ds_id)
        self.ckan.upsert(df, ds_id, res_id)

        logger.info("Pipeline completed successfully")
