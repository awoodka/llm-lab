"""Push run bundles to the showcase site on the web host.

Settings come from env (LAB_WEB_URL, LAB_INGEST_TOKEN) or lab/settings.yaml:
    web_url: https://web.example-tailnet.ts.net   # the site's publishing API, tailnet only
    ingest_token: <secret>
"""

import json
import os
import re
import sys
import time

import httpx
import yaml

from lab import paths
from lab.store import RunDir


class PublishError(RuntimeError):
    pass


# Published data names files, never paths on ai (`display_command()` shows weights by file name).
LOCAL_PATH_RE = re.compile(r"(?<![\w.:/-])/(?:home|mnt|srv|root)/")


def check_no_local_paths(payload: dict) -> None:
    text = json.dumps(payload)
    if m := LOCAL_PATH_RE.search(text):
        excerpt = text[max(0, m.start() - 30) : m.end() + 40]
        raise PublishError(f"bundle contains a local path ({excerpt!r}); publish file names only")


# Credentials a launch command or env dict could carry. The ingest token is sent as a header, never in the bundle.
SECRET_RE = re.compile(r"VLLM_API_KEY=\S|--api-key[= ]\S|Bearer\s+[\w.~+/-]{8,}|\bhf_[A-Za-z0-9]{20,}|\bsk-[A-Za-z0-9_-]{16,}")


def check_no_secrets(payload: dict) -> None:
    text = json.dumps(payload)
    if m := SECRET_RE.search(text):
        raise PublishError(f"bundle looks like it carries a credential near {text[max(0, m.start() - 30) : m.start() + 12]!r}…; remove it")


def settings() -> tuple[str, str]:
    file = yaml.safe_load(paths.SETTINGS.read_text()) if paths.SETTINGS.is_file() else {}
    url = os.environ.get("LAB_WEB_URL") or file.get("web_url")
    token = os.environ.get("LAB_INGEST_TOKEN") or file.get("ingest_token")
    if not url or not token:
        raise PublishError(f"set web_url and ingest_token in {paths.SETTINGS} (or LAB_WEB_URL / LAB_INGEST_TOKEN)")
    return url.rstrip("/"), token


def _client() -> tuple[httpx.Client, str]:
    url, token = settings()
    return httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=60), url


def publish(rd: RunDir, dry_run: bool = False, replace: bool = False) -> dict:
    bundle = rd.load()
    if bundle.run.status != "ok":
        raise PublishError(f"run {rd.path.name} has status {bundle.run.status!r}; only ok runs can be published")
    if bundle.run.raw.get("limited"):
        raise PublishError(
            f"run {rd.path.name} ran a --limit subset; it is a smoke test, not the pinned benchmark, so it stays local"
        )
    if bundle.run.kind == "evals":
        from lab.evals.registry import missing_from_tier

        if missing := missing_from_tier(bundle.run.tier, {r.task for r in bundle.eval_results}):
            raise PublishError(
                f"run {rd.path.name} has no {', '.join(missing)}; the site scores a tier as one run, so resume it "
                f"with --benchmarks {','.join(missing)} first"
            )
    payload = bundle.model_dump(mode="json")
    payload["bundle_sha"] = bundle.content_sha()
    check_no_local_paths(payload)
    check_no_secrets(payload)
    if dry_run:
        return {"dry_run": True, "run_id": bundle.run.id, "metrics": len(bundle.metrics), "bytes": len(str(payload))}
    client, url = _client()
    with client:
        r = client.post(f"{url}/api/ingest", params={"replace": "1"} if replace else None, json=payload)
    if r.status_code >= 400:
        raise PublishError(f"ingest failed ({r.status_code}): {r.text}")
    bundle.run.published_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    rd.save(bundle)
    return r.json()


def unpublish(rd: RunDir) -> None:
    bundle = rd.load()
    client, url = _client()
    with client:
        r = client.delete(f"{url}/api/runs/{bundle.run.id}")
    if r.status_code >= 400 and r.status_code != 404:
        raise PublishError(f"unpublish failed ({r.status_code}): {r.text}")
    bundle.run.published_at = None
    rd.save(bundle)


def put_hosted(config_hash: str | None) -> None:
    client, url = _client()
    with client:
        r = client.put(f"{url}/api/hosted", json={"config_hash": config_hash})
    if r.status_code >= 400:
        raise PublishError(f"hosted update failed ({r.status_code}): {r.text}")


PAUSE_TIMEOUT = httpx.Timeout(1.0)


def put_pause(paused: bool, reason: str | None = None, ref: str | None = None) -> bool:
    """Tell the site why the hosted chat model is down, or that the pause is over.

    GPU jobs call this, so it never raises and gives up after about a second. A lost report is harmless:
    the site probes the model's health itself and only uses this to explain why it's down.
    """
    try:
        url, token = settings()
    except Exception:  # noqa: BLE001
        return False
    try:
        r = httpx.put(
            f"{url}/api/hosted/pause",
            json={"paused": paused, "reason": reason, "ref": ref},
            headers={"Authorization": f"Bearer {token}"},
            timeout=PAUSE_TIMEOUT,
        )
        r.raise_for_status()
        return True
    except Exception as e:  # noqa: BLE001
        print(f"note: site status not updated ({type(e).__name__})", file=sys.stderr)
        return False
