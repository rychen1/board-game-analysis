"""BoardGameGeek XML API2 client. Returns raw artifacts only."""

from __future__ import annotations

import time
from datetime import UTC, datetime

import httpx

from board_game_analysis.config import Settings
from board_game_analysis.ingestion.bgg.parser import split_item_documents
from board_game_analysis.ingestion.bgg.types import RawArtifact
from board_game_analysis.ingestion.errors import (
    BggAuthenticationError,
    BggHttpError,
    BggNotFoundError,
)
from board_game_analysis.ingestion.util import chunked

SOURCE = "boardgamegeek"
RETRY_STATUSES = {202, 429, 503}
MAX_THING_IDS_PER_REQUEST = 20


class BggClient:
    """HTTP client for `/thing`. One request at a time; no HTML scraping."""

    def __init__(
        self,
        settings: Settings,
        http_client: httpx.Client | None = None,
    ) -> None:
        self._settings = settings
        self._owns_client = http_client is None
        self._http = http_client or httpx.Client(
            timeout=settings.bgg_timeout_seconds,
            headers=self._headers(),
        )
        self._last_request_at: float | None = None

    def close(self) -> None:
        if self._owns_client:
            self._http.close()

    def __enter__(self) -> BggClient:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def fetch_game(self, source_id: str) -> RawArtifact:
        artifacts = self.fetch_games([source_id])
        if not artifacts:
            msg = f"BGG thing id {source_id} was not found"
            raise BggNotFoundError(msg)
        return artifacts[0]

    def fetch_games(self, source_ids: list[str]) -> list[RawArtifact]:
        """Fetch thing ids in batches of at most 20. Missing ids are omitted."""
        token = self._require_token()
        artifacts: list[RawArtifact] = []
        batch_size = self._batch_size()
        for chunk in chunked(source_ids, batch_size):
            artifacts.extend(self._fetch_chunk(chunk, token))
        return artifacts

    def _require_token(self) -> str:
        token = self._settings.bgg_token
        if not token:
            msg = (
                "BGG XML API2 requires an application token. Register at "
                "https://boardgamegeek.com/applications and set BGG_TOKEN. "
                "Do not scrape HTML."
            )
            raise BggAuthenticationError(msg)
        return token

    def _batch_size(self) -> int:
        return min(self._settings.bgg_batch_size, MAX_THING_IDS_PER_REQUEST)

    def _fetch_chunk(self, source_ids: list[str], token: str) -> list[RawArtifact]:
        url = self._thing_url(source_ids)
        response = self._request(url, token)
        if response.status_code == 401 or response.status_code == 403:
            msg = (
                f"BGG returned {response.status_code}. Check BGG_TOKEN. "
                "See https://boardgamegeek.com/using_the_xml_api"
            )
            raise BggAuthenticationError(msg)
        if response.status_code >= 400:
            msg = f"BGG HTTP {response.status_code} for {url}"
            raise BggHttpError(msg)
        retrieved_at = datetime.now(UTC)
        content_type = response.headers.get("content-type", "application/xml")
        content_type = content_type.split(";")[0].strip()
        documents = split_item_documents(response.text)
        artifacts: list[RawArtifact] = []
        for source_id in source_ids:
            body = documents.get(source_id)
            if body is None:
                continue
            artifacts.append(
                RawArtifact(
                    source=SOURCE,
                    source_identifier=str(source_id),
                    retrieved_at=retrieved_at,
                    request_url=url,
                    content_type=content_type,
                    http_status=response.status_code,
                    body=body,
                )
            )
        return artifacts

    def _headers(self) -> dict[str, str]:
        return {"User-Agent": self._settings.http_user_agent}

    def _thing_url(self, source_ids: list[str] | str) -> str:
        if isinstance(source_ids, str):
            joined = source_ids
        else:
            joined = ",".join(source_ids)
        base = self._settings.bgg_base_url.rstrip("/")
        return f"{base}/thing?id={joined}&stats=1"

    def _request(self, url: str, token: str) -> httpx.Response:
        last_error: BggHttpError | None = None
        for attempt in range(self._settings.bgg_max_retries):
            self._respect_rate_limit()
            response = self._http.get(
                url,
                headers={
                    **self._headers(),
                    "Authorization": f"Bearer {token}",
                },
            )
            self._last_request_at = time.monotonic()
            if response.status_code not in RETRY_STATUSES:
                return response
            last_error = BggHttpError(
                f"BGG HTTP {response.status_code} for {url} (attempt {attempt + 1})"
            )
            time.sleep(self._settings.bgg_retry_backoff_seconds * (attempt + 1))
        if last_error is None:
            msg = f"BGG request failed for {url}"
            raise BggHttpError(msg)
        raise last_error

    def _respect_rate_limit(self) -> None:
        delay = self._settings.bgg_rate_limit_seconds
        if delay <= 0 or self._last_request_at is None:
            return
        elapsed = time.monotonic() - self._last_request_at
        remaining = delay - elapsed
        if remaining > 0:
            time.sleep(remaining)
