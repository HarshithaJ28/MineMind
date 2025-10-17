"""Simple filesystem-backed job queue.

Jobs are stored under `jobs/<job_id>/job.json` with status fields. This keeps the implementation lightweight
and avoids external dependencies. The worker polls for queued jobs and executes them.
"""
from __future__ import annotations
import os
import json
import uuid
import time
from datetime import datetime
from typing import List, Dict, Any

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
JOBS_DIR = os.path.join(ROOT, 'jobs')
os.makedirs(JOBS_DIR, exist_ok=True)


def _job_dir(job_id: str) -> str:
    return os.path.join(JOBS_DIR, job_id)


def enqueue_job(command: List[str], metadata: Dict[str, Any] | None = None) -> str:
    job_id = datetime.utcnow().strftime('%Y%m%dT%H%M%SZ') + '_' + uuid.uuid4().hex[:8]
    jd = _job_dir(job_id)
    os.makedirs(jd, exist_ok=True)
    job = {
        'id': job_id,
        'command': command,
        'metadata': metadata or {},
        'status': 'queued',
        'created_at': datetime.utcnow().isoformat() + 'Z',
        'started_at': None,
        'finished_at': None,
        'returncode': None,
    }
    with open(os.path.join(jd, 'job.json'), 'w') as f:
        json.dump(job, f, indent=2)
    # create empty log
    open(os.path.join(jd, 'log.txt'), 'a').close()
    return job_id


def list_jobs() -> List[Dict[str, Any]]:
    out = []
    for name in sorted(os.listdir(JOBS_DIR), reverse=True):
        jd = os.path.join(JOBS_DIR, name)
        jf = os.path.join(jd, 'job.json')
        if os.path.exists(jf):
            with open(jf, 'r') as f:
                try:
                    out.append(json.load(f))
                except Exception:
                    continue
    return out


def read_job(job_id: str) -> Dict[str, Any] | None:
    jf = os.path.join(_job_dir(job_id), 'job.json')
    if os.path.exists(jf):
        with open(jf, 'r') as f:
            return json.load(f)
    return None


def update_job(job_id: str, **patch: Any) -> None:
    jf = os.path.join(_job_dir(job_id), 'job.json')
    if not os.path.exists(jf):
        raise FileNotFoundError(jf)
    with open(jf, 'r') as f:
        job = json.load(f)
    job.update(patch)
    with open(jf, 'w') as f:
        json.dump(job, f, indent=2)


def job_log_path(job_id: str) -> str:
    return os.path.join(_job_dir(job_id), 'log.txt')


def read_log(job_id: str, max_bytes: int = 200_000) -> str:
    p = job_log_path(job_id)
    if not os.path.exists(p):
        return ''
    with open(p, 'rb') as f:
        f.seek(0, 2)
        size = f.tell()
        start = max(0, size - max_bytes)
        f.seek(start)
        data = f.read().decode('utf-8', errors='replace')
    return data
