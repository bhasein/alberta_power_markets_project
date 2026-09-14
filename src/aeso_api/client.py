"""Shared authenticated HTTP client with retries and atomic raw persistence."""

from __future__ import annotations

import hashlib
import json
import os
import ssl
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import certifi

from config import PROJECT_ROOT


BASE_URL = "https://apimgw.aeso.ca/public"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}


class AESORequestError(RuntimeError):
    """Raised when an AESO request fails or returns invalid JSON."""


@dataclass(frozen=True)
class DownloadResult:
    """Outcome of one raw API download."""

    path: Path
    status: str
    bytes_written: int


def validate_aeso_payload(payload: Any) -> None:
    """Validate the common AESO response envelope when it is present."""

    if not isinstance(payload, dict) or "responseCode" not in payload:
        return
    if str(payload["responseCode"]) != "200":
        raise AESORequestError(
            "AESO returned a non-success responseCode inside an HTTP 200 "
            f"response: {payload['responseCode']!r}."
        )
    if "return" not in payload:
        raise AESORequestError(
            "AESO success response did not contain the expected 'return' field."
        )


def read_project_env_value(name: str) -> str | None:
    """Read one value from the ignored project-root .env file."""

    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return None

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != name:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        return value or None

    return None


class AESOClient:
    """Minimal client for AESO public JSON APIs."""

    def __init__(
        self,
        api_key: str | None = None,
        *,
        timeout_seconds: int = 120,
        maximum_attempts: int = 4,
    ) -> None:
        self.api_key = (
            api_key
            or os.environ.get("AESO_API_KEY")
            or read_project_env_value("AESO_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Set AESO_API_KEY in the environment or project-root .env "
                "before downloading."
            )
        self.timeout_seconds = timeout_seconds
        self.maximum_attempts = maximum_attempts
        self.ssl_context = ssl.create_default_context(cafile=certifi.where())

    def _request(self, path: str, params: dict[str, str]) -> tuple[bytes, str]:
        url = f"{BASE_URL}{path}?{urlencode(params)}"
        request = Request(
            url,
            headers={
                "API-KEY": self.api_key,
                "Accept": "application/json",
                "User-Agent": "alberta-power-markets/0.1",
            },
            method="GET",
        )

        for attempt in range(1, self.maximum_attempts + 1):
            try:
                with urlopen(
                    request,
                    timeout=self.timeout_seconds,
                    context=self.ssl_context,
                ) as response:
                    body = response.read()
                    content_type = response.headers.get("Content-Type", "")
                try:
                    payload = json.loads(body.decode("utf-8"))
                except (UnicodeDecodeError, json.JSONDecodeError) as error:
                    raise AESORequestError(
                        f"AESO returned invalid JSON for {url}."
                    ) from error
                validate_aeso_payload(payload)
                return body, content_type
            except HTTPError as error:
                details = error.read().decode("utf-8", errors="replace")
                if (
                    error.code in RETRYABLE_STATUS_CODES
                    and attempt < self.maximum_attempts
                ):
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise AESORequestError(
                    f"AESO request failed with HTTP {error.code}: {details}"
                ) from error
            except (TimeoutError, URLError) as error:
                if attempt < self.maximum_attempts:
                    time.sleep(2 ** (attempt - 1))
                    continue
                raise AESORequestError(f"AESO request failed: {error}") from error

        raise AESORequestError("AESO request failed without a response.")

    def download_json(
        self,
        path: str,
        params: dict[str, str],
        output_path: Path,
        *,
        overwrite: bool = False,
    ) -> DownloadResult:
        """Download one response and save raw JSON plus a provenance sidecar."""

        output_path = output_path.resolve()
        metadata_path = output_path.with_suffix(".metadata.json")

        if output_path.exists() and not overwrite:
            try:
                payload = json.loads(output_path.read_text(encoding="utf-8"))
                validate_aeso_payload(payload)
            except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AESORequestError(
                    f"Existing raw file is invalid; inspect or use --overwrite: "
                    f"{output_path}"
                ) from error
            return DownloadResult(output_path, "already_exists", output_path.stat().st_size)

        body, content_type = self._request(path, params)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        part_path = output_path.with_suffix(output_path.suffix + ".part")
        part_path.write_bytes(body)
        part_path.replace(output_path)

        metadata = {
            "retrieved_at_utc": datetime.now(UTC).isoformat(),
            "request_url": f"{BASE_URL}{path}",
            "request_parameters": params,
            "content_type": content_type,
            "response_bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        metadata_part = metadata_path.with_suffix(metadata_path.suffix + ".part")
        metadata_part.write_text(
            json.dumps(metadata, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        metadata_part.replace(metadata_path)

        return DownloadResult(output_path, "downloaded", len(body))
