from opensearchpy import OpenSearch

from .config import settings


def get_client() -> OpenSearch:
    """
    Build an OpenSearch client that works with the default docker-compose setup.
    """
    return OpenSearch(
        hosts=[settings.opensearch_host],
        http_auth=(settings.opensearch_user, settings.opensearch_password),
        use_ssl=True,
        verify_certs=False,
        ssl_show_warn=False,
    )
