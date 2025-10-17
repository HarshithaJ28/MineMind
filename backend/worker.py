"""Worker process to execute filesystem-enqueued jobs from backend.job_queue.

Start this in a separate terminal: python backend/worker.py
It will poll jobs/ and run any with status 'queued'.
"""
from __future__ import annotations
import subprocess
import time
import os
import sys
# Ensure project root is on sys.path when running worker as a script
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from backend.job_queue import list_jobs, read_job, update_job, job_log_path, _job_dir


def main(poll_interval: float = 2.0):
    print('Worker started. Polling for jobs...')
    while True:
        try:
            jobs = list_jobs()
            for job in jobs:
                if job.get('status') == 'queued':
                    job_id = job['id']
                    jd = _job_dir(job_id)
                    logp = job_log_path(job_id)
                    update_job(job_id, status='running', started_at=time.strftime('%Y-%m-%dT%H:%M:%SZ'))
                    cmd = job['command']
                    print('Running job', job_id, cmd)
                    with open(logp, 'ab') as lf:
                        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf, cwd=os.getcwd())
                        rc = proc.wait()
                        update_job(job_id, status='finished', finished_at=time.strftime('%Y-%m-%dT%H:%M:%SZ'), returncode=rc)
                        print('Finished job', job_id, 'rc=', rc)
        except Exception as e:
            print('Worker error:', e)
        time.sleep(poll_interval)


if __name__ == '__main__':
    main()
