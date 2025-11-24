# CKAN Portal Metrics Plugin
---
CKAN Portal Metrics Plugin
A CKAN extension for exporting Google Analytics 4 (GA4) metrics and storing them in CKAN’s Datastore.
 
## Table of Contents

- [Service Account Creation and Permission Granting](#service-account-creation-and-permission-granting)
- [Configuration](#configuration)
- [How Configuration and Credentials Are Handled](#how-configuration-and-credentials-are-handled)
- [How the CLI Command Works](#how-the-cli-command-works)
- [Example Usage](#example-usage)
- [Best Practices](#best-practices)

## Service Account Creation and Permission Granting

To allow this plugin to access your Google Analytics 4 (GA4) data, you must create a Google Cloud service account and grant it the correct permissions.

### 1. Create a Service Account

1. Go to the [Google Cloud Console](https://console.cloud.google.com/).
2. Select or create a project.
3. Navigate to **IAM & Admin → Service Accounts**.
4. Click **Create Service Account**.
5. Enter a name and description, then click **Create and Continue**.
6. Assign the role **Viewer** (or a more restrictive custom role if desired).
7. Click **Done**.

### 2. Create and Download a JSON Key

1. In the Service Accounts list, click on your new service account.
2. Go to the **Keys** tab.
3. Click **Add Key → Create new key**.
4. Choose **JSON** and click **Create**.
5. Download the JSON file.  
   **Important:** Keep this file secure. You will need its contents for your CKAN config.

### 3. Grant Access to Your GA4 Property

1. Go to [Google Analytics Admin](https://analytics.google.com/analytics/web/).
2. Select your GA4 property.
3. Under **Property**, click **Account Access Management** or **Property Access Management**.
4. Click the **+** button and choose **Add users**.
5. Enter the **service account email** (from your JSON key).
6. Assign at least the **Viewer** role (or higher as needed).
7. Click **Add**.

### 4. Add Credentials to CKAN Config

Copy the relevant fields from your JSON key into your `ckan.ini` as described above:
- `ckanext.portal_metrics.ga4_client_email`
- `ckanext.portal_metrics.ga4_private_key` (escape newlines as `\\n`)
- `ckanext.portal_metrics.ga4_token_uri`

---

**Your service account is now ready to be used by the CKAN Portal Metrics plugin.**

---
## Configuration

Add the following to your `ckan.ini` (or equivalent config):

```ini
ckan.plugins = ... portal_metrics
# Required: Google Analytics 4 service account credentials
ckanext.portal_metrics.ga4_client_email = your-service-account@project.iam.gserviceaccount.com
ckanext.portal_metrics.ga4_private_key = -----BEGIN PRIVATE KEY-----\\nMIIEv...IDAQAB\\n-----END PRIVATE KEY-----\\n
ckanext.portal_metrics.ga4_token_uri = https://oauth2.googleapis.com/token

# Required: Your GA4 property ID
ckanext.portal_metrics.ga4_property_id = 123456789

# No trailing slash!
ckan.site_url = https://your-portal.example.com
```
---

## How Configuration and Credentials Are Handled

- The plugin implements `IConfigurable` and reads all required config values at startup.
- It validates that all required values are present and raises an error if any are missing.
- Credentials and settings (`ga4_credentials`, `property_id`, `hostname`) are stored as class attributes on the plugin (`MetricsCliPlugin`).
- These attributes are then accessed by the CLI command and pipeline code.

---

## How the CLI Command Works

- The CLI group and command are registered via the plugin using `IClick`.
- When you run `ckan metrics fetch`, the CLI:
    - Gets the sysadmin user context.
    - Reads credentials, property ID, and hostname from the plugin’s class attributes.
    - Passes these to the `MetricsPipeline`, which runs the full ETL process.

**Example usage:**
```sh
ckan metrics fetch --lookback-days=7
```

---

## Best Practices

- All configuration is managed via CKAN config (`ckan.ini`).
- Credentials are loaded and validated once at startup.
- No secrets are hardcoded or loaded from files at runtime.
- The CLI and pipeline access config/credentials via the plugin’s class attributes.
---
