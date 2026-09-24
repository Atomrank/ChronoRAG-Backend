"""
End-to-end HTTP smoke runner for Kaalkram pipelines.

  python scripts/e2e.py path/to/book.pdf [--base-url http://127.0.0.1:8000]

Uploads the PDF, builds naive / kaalkram / kaalkram_v2, polls jobs, and prints
build stats plus build_events summary.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError as exc:  # pragma: no cover
    raise SystemExit("httpx is required: pip install httpx") from exc

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _poll(client: httpx.Client, job_id: str, *, timeout_s: float = 3600.0,
          interval_s: float = 2.0) -> dict:
    t0 = time.time()
    while True:
        r = client.get(f"/api/jobs/{job_id}")
        r.raise_for_status()
        row = r.json()
        status = row.get("status")
        prog = row.get("progress")
        stage = row.get("stage")
        print(f"  [{job_id[:8]}] {status}  {prog}  {stage}", flush=True)
        if status in ("done", "error"):
            return row
        if time.time() - t0 > timeout_s:
            raise TimeoutError(f"job {job_id} timed out after {timeout_s}s")
        time.sleep(interval_s)


def main() -> None:
    ap = argparse.ArgumentParser(description="E2E upload + build via HTTP API")
    ap.add_argument("pdf", type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--timeout", type=float, default=3600.0,
                    help="per-job poll timeout seconds")
    ap.add_argument("--kinds", nargs="+",
                    default=["naive", "kaalkram", "kaalkram_v2"],
                    choices=["naive", "kaalkram", "kaalkram_v2"])
    args = ap.parse_args()
    if not args.pdf.is_file():
        raise SystemExit(f"PDF not found: {args.pdf}")

    with httpx.Client(base_url=args.base_url, timeout=120.0) as client:
        health = client.get("/api/health")
        if health.status_code != 200:
            raise SystemExit(f"API not healthy at {args.base_url}: {health.status_code}")

        print(f"uploading {args.pdf} ...")
        with args.pdf.open("rb") as fh:
            r = client.post(
                "/api/documents",
                files={"file": (args.pdf.name, fh, "application/pdf")},
            )
        r.raise_for_status()
        doc = r.json()
        doc_id = doc["id"]
        print(f"doc_id={doc_id}  title={doc.get('title')}  pages={doc.get('page_count')}")

        results = {}
        for kind in args.kinds:
            print(f"\n=== build {kind} ===")
            br = client.post(f"/api/documents/{doc_id}/build/{kind}")
            br.raise_for_status()
            job = br.json()
            done = _poll(client, job["id"], timeout_s=args.timeout)
            results[kind] = done
            if done.get("status") == "error":
                print(f"ERROR: {done.get('error')}")
            else:
                detail = done.get("detail") or {}
                print("build detail:")
                print(json.dumps(detail, indent=2, default=str)[:4000])
                be = detail.get("build_events")
                if be is not None:
                    print("build_events summary:", json.dumps(be, indent=2, default=str))

        # Document-level build_events (all jobs)
        try:
            mr = client.get(f"/api/documents/{doc_id}/metrics")
            if mr.status_code == 200:
                m = mr.json()
                if "build_events" in m:
                    print("\n=== document build_events ===")
                    print(json.dumps(m["build_events"], indent=2, default=str))
        except Exception as exc:
            print(f"(metrics fetch skipped: {exc})")

        print("\n=== summary ===")
        for kind, row in results.items():
            print(f"{kind}: status={row.get('status')}  "
                  f"error={row.get('error')!r}")
        print(f"doc_id={doc_id}")


if __name__ == "__main__":
    main()
