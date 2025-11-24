import logging
from urllib.parse import urlparse
from google.oauth2 import service_account
from ckan.plugins import SingletonPlugin, implements, IClick, IConfigurable


log = logging.getLogger(__name__)


class MetricsCliPlugin(SingletonPlugin):
    """
    CKAN plugin to register the metrics CLI command.
    """
    implements(IClick)
    implements(IConfigurable)

    # Class-level storage for credentials
    ga4_credentials = None
    property_id = None
    hostname = None

    def get_commands(self):
        from ckanext.portal_metrics.cli.metrics import metrics
        return [metrics]

    def configure(self, ckan_config):
        """
        Reads and validates GA4 service account credentials from CKAN config.
        Returns a google.oauth2.service_account.Credentials object.
        Raises Exception if any required config is missing.
            """
        self.property_id = ckan_config.get('ckanext.portal_metrics.ga4_property_id')
        url = ckan_config.get('ckan.site_url')

        self.hostname = urlparse(url).hostname
        client_email = ckan_config.get('ckanext.portal_metrics.ga4_client_email')
        private_key = ckan_config.get('ckanext.portal_metrics.ga4_private_key')
        token_uri = ckan_config.get('ckanext.portal_metrics.ga4_token_uri', 'https://oauth2.googleapis.com/token')

        missing = []
        if not self.property_id:
            missing.append('ckanext.portal_metrics.ga4_property_id')
        if not client_email:
            missing.append('ckanext.portal_metrics.ga4_client_email')
        if not private_key:
            missing.append('ckanext.portal_metrics.ga4_private_key')
        if not token_uri:
            missing.append('ckanext.portal_metrics.ga4_token_uri')

        if missing:
            log.error("Missing GA4 service account config values: %s", ', '.join(missing))
            return

        # Fix escaped newlines in private key
        private_key = private_key.replace('\\n', '\n')

        service_account_info = {
            "type": "service_account",
            "client_email": client_email,
            "private_key": private_key,
            "token_uri": token_uri,
        }
        try:
            self.ga4_credentials = service_account.Credentials.from_service_account_info(service_account_info)
        except Exception as e:
            log.error("Failed to create GA4 credentials: %s", e)
