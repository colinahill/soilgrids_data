"""Plain-S3 operations on the Source Coop product that icechunk does not cover:

- ``product/README.md`` -> the product root, outside the store prefix: this is
  the landing page source.coop renders
- audit artifacts (source manifest, validation reports) -> ``audit/{version}/``
- wiping a store prefix, to abandon a version path

data.source.coop does not implement the batch ``DeleteObjects`` operation (it
answers with a misleading ``NoSuchBucket``), so deletes here are per-key.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import boto3
import botocore.session
from botocore.client import Config
from botocore.credentials import RefreshableCredentials
from botocore.exceptions import ClientError
from tqdm import tqdm

from . import store as store_config

log = logging.getLogger(__name__)

PRODUCT_README = Path("product/README.md")
CREDENTIAL_ERROR_CODES = ("ExpiredToken", "InvalidToken", "TokenRefreshRequired", "AccessDenied")

CONTENT_TYPES = {
    ".md": "text/markdown",
    ".json": "application/json",
    ".parquet": "application/vnd.apache.parquet",
    ".csv": "text/csv",
}


class CredentialError(RuntimeError):
    """Credentials are dead; needs a human (`source-coop login`), not a retry."""


def _creds_metadata(credentials_file: str | None) -> dict:
    """Credentials in botocore RefreshableCredentials metadata form.

    The advertised expiry makes botocore re-resolve credentials periodically, so
    an operation that outlives one credential set picks up a refreshed
    creds.json / source-coop CLI login instead of dying on ExpiredToken.
    """
    creds = store_config.load_credentials(credentials_file)
    # cap the reported expiry so re-resolution happens at least every ~15 min
    # (botocore refreshes ~15 min before the advertised time)
    cap = datetime.now(UTC) + timedelta(minutes=30)
    expiry = min(datetime.fromisoformat(creds["expires_at"]), cap) if creds.get("expires_at") else cap
    return {
        "access_key": creds["access_key_id"],
        "secret_key": creds["secret_access_key"],
        "token": creds["session_token"],
        "expiry_time": expiry.isoformat(),
    }


def client(credentials_file: str | None = None):
    refreshable = RefreshableCredentials.create_from_metadata(
        metadata=_creds_metadata(credentials_file),
        refresh_using=lambda: _creds_metadata(credentials_file),
        method="source-coop",
    )
    session = botocore.session.get_session()
    session._credentials = refreshable
    return boto3.Session(botocore_session=session).client(
        "s3",
        endpoint_url=store_config.SOURCE_COOP_ENDPOINT,
        region_name=store_config.SOURCE_COOP_REGION,
        config=Config(
            s3={"addressing_style": "path"},  # no wildcard DNS for virtual-hosted style
            # not "adaptive": its client-side rate limiter throttles every worker
            # to a crawl after a few 5xx/429s, which looks like a hang
            retries={"max_attempts": 10, "mode": "standard"},
            connect_timeout=10,
            read_timeout=60,
        ),
    )


def _translate(exc: ClientError, what: str) -> RuntimeError:
    """One readable line, including the S3 error code (Rich truncates tracebacks)."""
    code = exc.response.get("Error", {}).get("Code", "unknown")
    message = f"{what} failed: {code}"
    log.error(message)
    if code in CREDENTIAL_ERROR_CODES:
        return CredentialError(f"{message} - credentials expired: run `source-coop login` (or refresh creds.json)")
    return RuntimeError(message)


def upload_readme(
    account: str,
    product: str = store_config.PRODUCT_NAME,
    *,
    credentials_file: str | None = None,
    readme: Path = PRODUCT_README,
) -> None:
    """Publish ``product/README.md`` as the product's landing page."""
    if not readme.exists():
        raise FileNotFoundError(f"{readme} not found (run from the repo root)")
    key = f"{product}/README.md"
    s3 = client(credentials_file)
    try:
        s3.put_object(Bucket=account, Key=key, Body=readme.read_bytes(), ContentType="text/markdown")
    except ClientError as exc:
        raise _translate(exc, f"README upload to s3://{account}/{key}") from None
    log.info("uploaded %s to s3://%s/%s", readme, account, key)


def upload_audit(
    account: str,
    files: list[Path],
    version: str,
    product: str = store_config.PRODUCT_NAME,
    *,
    credentials_file: str | None = None,
) -> None:
    """Upload audit artifacts (manifest, reports) under audit/{version}/."""
    s3 = client(credentials_file)
    for path in tqdm(files, desc="uploading audit artifacts", unit="file"):
        key = f"{product}/audit/{version}/{path.name}"
        content_type = CONTENT_TYPES.get(path.suffix, "application/octet-stream")
        try:
            s3.put_object(Bucket=account, Key=key, Body=path.read_bytes(), ContentType=content_type)
        except ClientError as exc:
            raise _translate(exc, f"audit upload to s3://{account}/{key}") from None
        log.info("uploaded %s to s3://%s/%s", path, account, key)


def store_keys(
    account: str,
    product: str = store_config.PRODUCT_NAME,
    *,
    credentials_file: str | None = None,
    s3=None,
) -> tuple[list[str], int]:
    """Every object key under the versioned store prefix, plus their total bytes."""
    s3 = s3 or client(credentials_file)
    prefix = f"{product}/{store_config.STORE_SUBPATH}/"
    keys, total = [], 0
    for page in s3.get_paginator("list_objects_v2").paginate(Bucket=account, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
            total += obj["Size"]
    return keys, total


def delete_keys(
    account: str, keys: list[str], *, credentials_file: str | None = None, workers: int = 8, s3=None
) -> None:
    """Delete keys one request at a time."""
    s3 = s3 or client(credentials_file)
    with (
        tqdm(total=len(keys), desc="deleting remote objects", unit="obj") as bar,
        ThreadPoolExecutor(max_workers=workers) as pool,
    ):
        futures = [pool.submit(s3.delete_object, Bucket=account, Key=key) for key in keys]
        try:
            for future in as_completed(futures):
                future.result()
                bar.update(1)
        except ClientError as exc:
            pool.shutdown(wait=False, cancel_futures=True)
            raise _translate(exc, "delete") from None
