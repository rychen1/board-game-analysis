"""Ingestion errors. These must not leak into the domain package."""


class IngestionError(Exception):
    """Base error for the ingestion layer."""


class BggAuthenticationError(IngestionError):
    """BGG XML API2 rejected the request or no token was configured."""


class BggNotFoundError(IngestionError):
    """The requested thing id was not present in the API response."""


class BggHttpError(IngestionError):
    """A non-success HTTP response after retries."""
