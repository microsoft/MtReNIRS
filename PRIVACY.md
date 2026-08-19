# Privacy

Last updated: August 19, 2026

## Scope

MtReNIRS is locally run research software. The project does not provide a
hosted service, require a user account, or implement project-specific
telemetry. Its Python APIs and command-line tools process data in the
environment where they are run. The project maintainers do not receive input
signals, sensor geometry, metadata, embeddings, or exported results.

## Local user interface

The optional MtSi1005 user interface is built with Streamlit. When the
interface is run locally, files selected in the browser are handled by the
local Streamlit process. Results are written only when the user requests an
export.

Streamlit is a third-party framework. Its `browser.gatherUsageStats` setting
defaults to `true`, which may send framework-level usage statistics to
Streamlit. MtReNIRS does not add to or receive those statistics. To disable
them, set the following environment variable before starting the interface:

```bash
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
mtsi1005-ui
```

Alternatively, place the following in `.streamlit/config.toml` in the
directory from which the interface is launched:

```toml
[browser]
gatherUsageStats = false
```

See the [Streamlit configuration documentation](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)
and [Streamlit privacy notice](https://www.snowflake.com/en/legal/privacy/privacy-policy/)
for details about that third-party behavior.

## Remote deployments

If someone deploys the user interface on a remote server, input files and
browser traffic are processed by that deployment. The deployment operator is
responsible for access controls, retention, security, and any additional
privacy notice required for that service. Do not submit sensitive or personal
data to a deployment unless you trust its operator and understand its terms.

For information about Microsoft privacy practices, see the
[Microsoft Privacy Statement](https://go.microsoft.com/fwlink/?LinkId=521839).