import re
import time
import logging
from datetime import date, timedelta

import pandas as pd
from google.analytics.data_v1beta import (
    BetaAnalyticsDataClient,
    FilterExpression,
    FilterExpressionList,
)
from google.analytics.data_v1beta.types import (
    DateRange,
    Dimension,
    Metric,
    RunReportRequest,
    Filter
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

    ANALYTICS_DIMENSIONS = [
        "date", "pagePath", "pageTitle", "hostName",
    ]
    ANALYTICS_METRICS = [
        "screenPageViews", "totalUsers", "userEngagementDuration",
        "activeUsers", "averageSessionDuration", "newUsers",
    ]
    DOWNLOADS_DIMENSIONS = [
        "date", "pagePath", "pageTitle", "hostName", "fileName", "fileExtension"
    ]
    DOWNLOADS_METRICS = [
        "eventCount"
    ]

    def __init__(self, creds, property_id, hostname):
        self.property_id = property_id
        self.hostname = hostname
        self.client = BetaAnalyticsDataClient(credentials=creds)

    def _fetch_analytics(self, start_date, end_date, offset, limit):
        return RunReportRequest(
            property=f"properties/{self.property_id}",
            dimensions=[Dimension(name=n) for n in self.ANALYTICS_DIMENSIONS],
            metrics=[Metric(name=m) for m in self.ANALYTICS_METRICS],
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

    def _fetch_downloads(self, start_date, end_date, offset, limit):
        return RunReportRequest(
            property=f"properties/{self.property_id}",
            dimensions=[Dimension(name=n) for n in self.DOWNLOADS_DIMENSIONS],
            metrics=[Metric(name=m) for m in self.DOWNLOADS_METRICS],
            date_ranges=[DateRange(start_date=start_date, end_date=end_date)],
            dimension_filter=FilterExpression(
                and_group=FilterExpressionList(
                    expressions=[
                        # eventName == file_download
                        FilterExpression(
                            filter=Filter(
                                field_name="eventName",
                                string_filter=Filter.StringFilter(value="file_download")
                            )
                        ),
                        # fileExtension NOT empty
                        FilterExpression(
                            not_expression=FilterExpression(
                                filter=Filter(
                                    field_name="fileExtension",
                                    empty_filter=Filter.EmptyFilter()
                                )
                            )
                        ),
                        # hostName == your hostname
                        FilterExpression(
                            filter=Filter(
                                field_name="hostName",
                                string_filter=Filter.StringFilter(
                                    value=self.hostname,
                                    match_type=Filter.StringFilter.MatchType.EXACT,
                                    case_sensitive=False
                                )
                            )
                        ),
                    ]
                )
            ),
            offset=offset,
            limit=limit,
        )

    def _fetch_page(self, start_date, end_date, offset, limit, func):
        """
        Fetch a single page of results from GA4 with retry logic and error logging.
        """
        req = func(start_date, end_date, offset, limit)

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

    def _fetch_all(self, start_date, end_date, func, page_size=MAX_GA_PAGE_SIZE):
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
            resp = self._fetch_page(start_date, end_date, offset, page_size, func)
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
            return pd.DataFrame(columns=self.ANALYTICS_DIMENSIONS + self.ANALYTICS_METRICS)

    def fetch_analytics(self, start_date, end_date):
        return self._fetch_all(start_date, end_date, self._fetch_analytics)

    def fetch_downloads(self, start_date, end_date):
        return self._fetch_all(start_date, end_date, self._fetch_downloads)

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
        "newUsers", "activeUsers", "eventCount"
    ]
    ANALYTICS_AGG_MAP = {
        "group_by": ["date", "page_path"],
        "user_engagement_duration": ("userEngagementDuration", "sum"),
        "total_users": ("totalUsers", "sum"),
        "new_users": ("newUsers", "sum"),
        "screen_page_views": ("screenPageViews", "sum"),
        "active_users": ("activeUsers", "sum"),
        "avg_session_duration": ("averageSessionDuration", "mean"),
        "page_title": ("pageTitle", lambda s: " | ".join(sorted({x for x in s if pd.notna(x) and x}))),
    }
    DOWNLOADS_AGG_MAP = {
        "group_by": ["date", "file_name"],
        "downloads": ("eventCount", "sum"),
        "page_title": ("pageTitle", lambda s: " | ".join(sorted({x for x in s if pd.notna(x) and x}))),
        "file_extension": ("fileExtension", lambda s: " | ".join(sorted({x for x in s if pd.notna(x) and x}))),
        "page_path": ("pagePath", lambda s: " | ".join(sorted({x for x in s if pd.notna(x) and x}))),
    }
    RE_VIEW = re.compile(r"/view/.*$")
    RE_RESOURCE = re.compile(r"((?:/resource/[a-f0-9-]+))/.*")

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
        df["page_path"] = (df.get("pagePath", "").fillna("").str.strip())
        return df

    def aggregate(self, df, aggr_map):
        # only keep aggregations for available columns
        group_by = aggr_map.pop("group_by")
        agg = {k: v for k, v in aggr_map.items() if v[0] in df.columns}
        df = df.groupby(group_by, dropna=False)
        df = df.agg(**agg).reset_index()
        return df


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
                groups = [group.get('display_name', '') for group in pkg.get("groups", [])]

                for res in pkg.get("resources", []):
                    self._cache[res["id"]] = {
                        "org_id": org_id,
                        "org_name": org_name,
                        "format": res.get("format", ""),
                        "resource_name": res.get("name", ""),
                        "groups": '; '.join(groups),
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
    MAX_CKAN_UPSERT_BATCH_SIZE = 5000
    ANALYTICS_DATA_DICT = [
        {"id": "date",
         "type": "date",
         "info": {
             "label": "Date",
             "notes": "GA4 calendar date (property timezone) for the aggregated metrics."}},
        {"id": "page_path",
         "type": "text",
         "info": {
             "label": "Page path",
             "notes": "Normalized CKAN page path used as the aggregation key (e.g. /dataset/ /resource/ )."}},
        {"id": "resource_id", "type": "text",
         "info": {
             "label": "Resource ID",
             "notes": "CKAN resource UUID extracted from page_path (/resource/ )."}         },
        {"id": "resource_name",
         "type": "text",
         "info": {"label": "Resource name",
                  "notes": "CKAN resource title at the time of ingestion."}},
        {"id": "user_engagement_duration",
         "type": "int",
         "info": {"label": "User engagement duration (s)",
                  "notes": "Sum of GA4 userEngagementDuration in seconds for the group (date + page_path)."}},
        {"id": "total_users",
         "type": "int",
         "info": {"label": "Total users",
                  "notes": "Distinct users (GA4 totalUsers) for the group."}},
        {"id": "new_users",
         "type": "int",
         "info": {"label": "New users",
                  "notes": "First-time users (GA4 newUsers) for the group."}},
        {"id": "screen_page_views",
         "type": "int",
         "info": {"label": "Page views",
                  "notes": "Total page/screen views (GA4 screenPageViews) for the group."}},
        {"id": "active_users",
         "type": "int",
         "info": {"label": "Active users",
                  "notes": "Active users (GA4 activeUsers) for the group."}},
        {"id": "avg_session_duration",
         "type": "float",
         "info": {"label": "Avg. session duration (s)",
                  "notes": "Mean of GA4 averageSessionDuration in seconds for the group."}},
        {"id": "page_title",
         "type": "text",
         "info": {"label": "Page title(s)",
                  "notes": "Unique page titles seen for the group; ‘|’-separated if multiple."}},
        {"id": "org_id",
         "type": "text",
         "info": {"label": "Organization ID",
                  "notes": "Owning CKAN organization ID."}},
        {"id": "org_name",
         "type": "text",
         "info": {"label": "Organization",
                  "notes": "Owning CKAN organization display name."}},
        {"id": "format",
         "type": "text",
         "info": {"label": "Format",
                  "notes": "CKAN resource format (e.g. CSV, JSON)."}},
        {"id": "groups",
         "type": "text",
         "info": {"label": "Groups",
                  "notes": "Semicolon-separated CKAN group display names the dataset belongs to."}},
    ]
    DOWNLOADS_DATA_DICT = [
        {"id": "date",
         "type": "date",
         "info": {"label": "Date",
                  "notes": "GA4 calendar date (property timezone) for download events."}},
        {"id": "file_name",
         "type": "text",
         "info": {"label": "File name",
                  "notes": "GA4 file_download event parameter fileName (e.g. myfile.csv)."}},
        {"id": "resource_id",
         "type": "text",
         "info": {"label": "Resource ID",
                  "notes": "CKAN resource UUID extracted from page_path (/resource/ )."}},
        {"id": "resource_name",
         "type": "text",
         "info": {"label": "Resource name",
                  "notes": "CKAN resource title at the time of ingestion."}},
        {"id": "page_path",
         "type": "text",
         "info": {"label": "Page path",
                  "notes": "Normalized CKAN path(s) where the download occurred; ‘|’-separated if multiple."}},
        {"id": "downloads",
         "type": "int",
         "info": {"label": "Downloads",
                  "notes": "Sum of GA4 eventCount for file_download events (grouped by date + file_name)."}},
        {"id": "file_extension",
         "type": "text",
         "info": {"label": "File extension",
                  "notes": "Unique file extensions observed; ‘|’-separated if multiple (e.g. csv)."}},
        {"id": "page_title",
         "type": "text",
         "info": {"label": "Page title(s)",
                  "notes": "Unique page titles for the download page(s); ‘|’-separated if multiple."}},
        {"id": "org_id",
         "type": "text",
         "info": {"label": "Organization ID",
                  "notes": "Owning CKAN organization ID."}},
        {"id": "org_name",
         "type": "text",
         "info": {"label": "Organization",
                  "notes": "Owning CKAN organization display name."}},
        {"id": "format",
         "type": "text",
         "info": {"label": "Format",
                  "notes": "CKAN resource format (e.g. CSV, JSON)."}},
        {"id": "groups",
         "type": "text",
         "info": {"label": "Groups",
                  "notes": "Semicolon-separated CKAN group display names the dataset belongs to."}},
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

    def upsert(self, df, dataset_id, name, datadict, pk, resource_id=None):
        records = df.to_dict(orient="records")
        if not resource_id:
            payload = {
                "resource": {
                    "package_id": dataset_id,
                    "name": name,
                    "format": "csv"
                },
                "fields": datadict,
                "primary_key": pk,
                "records": records[:5],
                "force": True
            }
            result = tk.get_action('datastore_create')(self.context, payload)
            resource_id = result["resource_id"]
            time.sleep(20)

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
        dfa = self.ga4.fetch_analytics(start_date, end_date)
        dfa = self.proc.clean(dfa)
        dfa = self.proc.aggregate(dfa, self.proc.ANALYTICS_AGG_MAP)

        dfd = self.ga4.fetch_downloads(start_date, end_date)
        dfd = self.proc.clean(dfd)
        dfd = dfd.rename(columns={"fileName": "file_name"})
        dfd = self.proc.aggregate(dfd, self.proc.DOWNLOADS_AGG_MAP)

        config = [
            ("Portal Analytics Data", dfa, self.ckan.ANALYTICS_DATA_DICT, ["date", "page_path"], "page_path"),
            ("Portal Downloads Data", dfd, self.ckan.DOWNLOADS_DATA_DICT, ["date", "file_name"], "file_name")
        ]
        for name, df, data_dict, pk, uuid_source in config:
            # 2) Enrich titles + org/format
            df["resource_id"] = df[uuid_source].str.extract(r"/resource/([0-9a-fA-F\-]{36})")[0]
            meta = df["resource_id"].map(self.cache.get).apply(pd.Series)
            df = pd.concat([df, meta], axis=1)
            # 3) Upsert to CKAN
            ds_id = self.ckan.ensure_dataset(self.owner_org)
            res_id = self.ckan.find_resource(ds_id, name)
            self.ckan.upsert(df, ds_id, name, data_dict, resource_id=res_id, pk=pk)

        logger.info("Pipeline completed successfully")
