"""Icechunk storage/repository helpers.

Ported near-verbatim from usda_gnatsgo: the refreshable-credentials machinery,
the CREDENTIALS_MARKER trick for telling dead credentials from transient object-
store errors, and the local/s3/Source-Coop target resolution all apply unchanged.

Supported store targets, all readable and writable:
- local path (development / staging)
- any ``s3://bucket/prefix`` (uses the standard AWS credential chain)
- Source Cooperative product prefix (helper that fills in the well-known bucket)

Source Coop access uses temporary scoped STS credentials, from the ``source-coop``
CLI's cached login or the product page's JSON export. ``get_credentials`` re-reads
them on every icechunk refresh, so a multi-hour backfill survives credential
rotation; ``from_env=True`` (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY /
AWS_SESSION_TOKEN) is the fallback when neither source is available.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import icechunk

from . import config

# Source Coop S3-compatible access: bucket = the account name, prefix = the
# product name. These values (and the region) come from the product's
# credentials page - that page is the ground truth if they ever change.
SOURCE_COOP_ENDPOINT = "https://data.source.coop"
SOURCE_COOP_REGION = "us-east-1"
SOURCE_COOP_ACCOUNT = "chill"
PRODUCT_NAME = "soilgrids"
# Versioned store directory: each ISRIC release (and any breaking structural
# change) is published as a fresh v{x.y.z}.icechunk path alongside older ones,
# so existing readers never break and old releases stay readable.
STORE_SUBPATH = f"v{config.DATASET_VERSION}.icechunk"


def source_coop_storage(
    account: str = SOURCE_COOP_ACCOUNT,
    product: str = PRODUCT_NAME,
    *,
    credentials_file: str | None = None,
    anonymous: bool = False,
) -> icechunk.Storage:
    """Icechunk storage for the Source Coop product (bucket=account, prefix=product)."""
    kwargs: dict = {
        "bucket": account,
        "prefix": f"{product}/{STORE_SUBPATH}",
        "region": SOURCE_COOP_REGION,
        "endpoint_url": SOURCE_COOP_ENDPOINT,
        "force_path_style": True,
    }
    if anonymous:
        kwargs["anonymous"] = True
    elif credentials_file or shutil.which("source-coop"):
        kwargs["get_credentials"] = _RefreshableCredentials(credentials_file)
    else:
        kwargs["from_env"] = True
    return icechunk.s3_storage(**kwargs)


DEFAULT_CREDS_FILE = "creds.json"

# icechunk calls get_credentials from Rust and re-raises whatever it gets as a
# generic StorageError with the Python message embedded, so a credential failure
# is indistinguishable by type from a transient object-store error. This marker
# is planted in the message we raise so callers can tell the two apart and stop
# instead of retrying; the S3-side strings cover credentials that expire between
# refreshes rather than at refresh time.
CREDENTIALS_MARKER = "soilgrids-credentials-unavailable"
_S3_CREDENTIAL_ERRORS = ("ExpiredToken", "InvalidAccessKeyId", "SignatureDoesNotMatch", "TokenRefreshRequired")


class CredentialsUnavailable(RuntimeError):
    """Credentials could not be resolved or were rejected: a human must act.

    A RuntimeError subclass because load_credentials' fallback chain already
    treats a RuntimeError as "try the next source".
    """

    def __init__(self, detail: str):
        super().__init__(f"{CREDENTIALS_MARKER}: {detail}")


def is_credentials_failure(exc: BaseException) -> bool:
    """Is this exception (or its message, after a round trip through icechunk)
    a credential problem that retrying cannot fix?"""
    if isinstance(exc, CredentialsUnavailable):
        return True
    if getattr(exc, "kind", None) == icechunk.ErrorKind.INVALID_CREDENTIALS:
        return True
    text = str(exc)
    return CREDENTIALS_MARKER in text or any(marker in text for marker in _S3_CREDENTIAL_ERRORS)


def _creds_from_file(path: str) -> dict[str, str | None]:
    """Source Coop's "JSON (SDK)" credential export format."""
    creds = json.loads(Path(path).read_text())
    return {
        "access_key_id": creds["aws_access_key_id"],
        "secret_access_key": creds["aws_secret_access_key"],
        "session_token": creds.get("aws_session_token"),
    }


def _creds_from_cli(cli: str) -> dict[str, str | None]:
    proc = subprocess.run([cli, "creds", "--format", "credential-process"], capture_output=True, text=True)
    try:
        creds = json.loads(proc.stdout)
    except json.JSONDecodeError:
        # the CLI exits 0 even without cached credentials; detect via non-JSON output
        message = (proc.stdout + proc.stderr).strip() or "no output"
        raise CredentialsUnavailable(f"source-coop creds failed ({message}); run `source-coop login`") from None
    return {
        "access_key_id": creds["AccessKeyId"],
        "secret_access_key": creds["SecretAccessKey"],
        "session_token": creds.get("SessionToken"),
        "expires_at": creds.get("Expiration"),  # ISO timestamp; lets callers track real expiry
    }


def load_credentials(credentials_file: str | None = None) -> dict[str, str | None]:
    """Source Coop credentials, in preference order:

    1. ``credentials_file`` if explicitly given and present
    2. the ``source-coop`` CLI's cached browser login (it tracks expiry, so a
       fresh ``source-coop login`` always wins over a stale export file)
    3. ``creds.json`` in the working directory
    """
    if credentials_file and Path(credentials_file).exists():
        return _creds_from_file(credentials_file)
    cli = shutil.which("source-coop")
    if cli:
        try:
            return _creds_from_cli(cli)
        except RuntimeError:
            if Path(DEFAULT_CREDS_FILE).exists():
                return _creds_from_file(DEFAULT_CREDS_FILE)
            raise
    if Path(DEFAULT_CREDS_FILE).exists():
        return _creds_from_file(DEFAULT_CREDS_FILE)
    raise CredentialsUnavailable(
        "no Source Coop credentials: run `source-coop login` (brew install "
        "source-cooperative/tap/source-coop), or save the product's JSON credential "
        f"export as {DEFAULT_CREDS_FILE} / pass --credentials-file."
    )


class _RefreshableCredentials:
    """get_credentials callable for icechunk that re-resolves credentials on
    each refresh, via load_credentials (JSON file or source-coop CLI cache).

    A module-level class (rather than a closure) because icechunk pickles the
    callable.
    """

    def __init__(self, credentials_file: str | None):
        self.credentials_file = credentials_file

    def __call__(self) -> icechunk.S3StaticCredentials:
        creds = load_credentials(self.credentials_file)
        return icechunk.S3StaticCredentials(
            access_key_id=creds["access_key_id"],
            secret_access_key=creds["secret_access_key"],
            session_token=creds["session_token"],
            expires_after=datetime.now(UTC) + timedelta(minutes=15),
        )


def storage_from_uri(
    uri: str,
    *,
    region: str = SOURCE_COOP_REGION,
    credentials_file: str | None = None,
    anonymous: bool = False,
) -> icechunk.Storage:
    """Resolve a store URI to icechunk Storage.

    ``uri`` is either a local path or ``s3://bucket/prefix``.
    """
    if uri.startswith("s3://"):
        bucket, _, prefix = uri.removeprefix("s3://").partition("/")
        kwargs: dict = {"bucket": bucket, "prefix": prefix, "region": region}
        if anonymous:
            kwargs["anonymous"] = True
        elif credentials_file:
            kwargs["get_credentials"] = _RefreshableCredentials(credentials_file)
        else:
            kwargs["from_env"] = True
        return icechunk.s3_storage(**kwargs)
    return icechunk.local_filesystem_storage(str(Path(uri).expanduser()))


def open_repo(storage: icechunk.Storage, *, create: bool = False) -> icechunk.Repository:
    if create:
        return icechunk.Repository.open_or_create(storage)
    return icechunk.Repository.open(storage)


class ConcurrentWriter(RuntimeError):
    """Another writer holds the branch and the two changesets genuinely overlap."""


def commit_with_rebase(session: icechunk.Session, message: str, *, tries: int = 100) -> str:
    """Commit, replaying our changes over anything that landed since we started.

    icechunk's branch commit is optimistic: ANY commit that lands between opening
    the session and committing makes ours fail, even when the two touched
    completely different arrays -- which is exactly the phase-3/phase-4 overlap
    (materializing one property while coarsening another). Rebase replays our
    changeset onto the new tip and only refuses when the changes really collide.

    The one collision that cannot be merged is two writers updating one node's
    metadata (`ZarrMetadataDoubleUpdate`): attrs are a single object, and
    ``BasicConflictSolver`` can only choose a whole version, never merge keys.
    That is why the overview resume token lives on the level array rather than on
    the shared group that ``materialize`` writes its cell ledger to.
    """
    try:
        return session.commit(message, rebase_with=icechunk.BasicConflictSolver(), rebase_tries=tries)
    except icechunk.RebaseFailedError as exc:
        where = ", ".join(f"{c.path} ({c.conflict_type})" for c in exc.conflicts[:4])
        raise ConcurrentWriter(
            f"another writer committed to main and the changes overlap at {where}; "
            f"let that run finish (or stop it) and re-run -- committed work is kept"
        ) from exc
