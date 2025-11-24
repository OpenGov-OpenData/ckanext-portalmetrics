import logging
import click
from ckan.plugins import toolkit

from ckanext.portal_metrics.pipeline import MetricsPipeline
from ckanext.portal_metrics.plugin import MetricsCliPlugin


log = logging.getLogger(__name__)


@click.group()
def metrics():
    """Portal Metrics commands."""


@metrics.command('fetch')
@click.option('--lookback-days', default=3, show_default=True, type=int,
              help='How many days back to fetch metrics')
@click.option('--owner-org', default=None, help='CKAN organization for dataset (default: portal-metrics)')
@click.option('--property-id-override', default=None, help='GA4 property_id (default: from ckan.ini)')
def fetch_metrics(lookback_days, owner_org, property_id_override):
    """
    Fetch Google Analytics metrics and upsert into CKAN datastore.
    """
    # Use CKAN config and sysadmin user
    owner_org = owner_org or 'portal-metrics'
    user = toolkit.get_action('get_site_user')({'ignore_auth': True}, {})
    context = {'user': user['name']}
    plugin = MetricsCliPlugin()
    creds = plugin.ga4_credentials
    hostname = plugin.hostname
    if property_id_override:
        property_id = property_id_override
    else:
        property_id = plugin.property_id
    log.info("Starting metrics pipeline")
    try:
        MetricsPipeline(property_id, hostname, context, creds, lookback_days, owner_org).run()
        log.info("Metrics pipeline completed successfully.")
    except Exception as e:
        log.exception("Metrics pipeline failed: %s", e)
        raise click.ClickException(str(e))
