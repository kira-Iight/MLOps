"""Замер нагрузки: 4+4 запроса при concurrency 1 и 2 (MPS-safe).

Сырые строки → metrics/load-raw.jsonl, сводка → docs/load-table.csv.
"""
import csv
import json
import shutil
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# Скрипт лежит в scripts/, остальные модули — в корне проекта
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def wait_ready(url, timeout=30):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(url + '/ready', timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.2)
    return False


def one_request(url, body):
    started = time.monotonic()
    try:
        req = urllib.request.Request(
            url + '/generate',
            data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            data = json.loads(r.read())
        return {
            'ok': True,
            'request_id': data.get('request_id'),
            'model_id': data.get('model_id'),
            'client_latency': round(time.monotonic() - started, 4),
            'generation_seconds': data.get('generation_seconds'),
            'output_tokens': data.get('output_tokens'),
            'error': None,
        }
    except Exception as exc:
        return {
            'ok': False,
            'request_id': None,
            'model_id': None,
            'client_latency': round(time.monotonic() - started, 4),
            'generation_seconds': None,
            'output_tokens': None,
            'error': f'{type(exc).__name__}: {exc}',
        }


def run_phase(url, concurrency, n, raw):
    body = {'prompt': 'Тест нагрузки: company earnings.', 'max_new_tokens': 8}
    started = time.monotonic()
    if concurrency == 1:
        results = [one_request(url, body) for _ in range(n)]
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            results = list(pool.map(lambda _: one_request(url, body), range(n)))
    total_seconds = time.monotonic() - started

    oks = [r for r in results if r['ok']]
    lats = [r['client_latency'] for r in oks]
    for r in results:
        r['concurrency'] = concurrency
        raw.append(r)

    return {
        'concurrency': concurrency,
        'n': n,
        'ok': len(oks),
        'errors': n - len(oks),
        'total_seconds': round(total_seconds, 3),
        'req_per_sec': round(len(oks) / total_seconds, 3) if total_seconds else 0,
        'mean_latency': round(sum(lats) / len(lats), 4) if lats else 0,
        'max_latency': round(max(lats), 4) if lats else 0,
    }


def main():
    ws = Path('.lab-load')
    if ws.exists():
        shutil.rmtree(ws)
    subprocess.run(
        [sys.executable, 'lab.py', '--workspace', str(ws), 'init'],
        check=True, cwd=Path(__file__).resolve().parents[1],
    )

    port = free_port()
    proc = subprocess.Popen(
        [sys.executable, 'service_fastapi.py', '--workspace', str(ws),
         '--port', str(port), '--warmup-seconds', '1', '--backend', 'stub'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd=Path(__file__).resolve().parents[1],
    )
    url = f'http://127.0.0.1:{port}'
    try:
        if not wait_ready(url):
            raise SystemExit('сервис не поднялся за 30 секунд')
        print(f'Сервис готов: {url}')

        raw = []
        table = [run_phase(url, 1, 4, raw)]
        print(f'concurrency=1: {table[-1]}')
        table.append(run_phase(url, 2, 4, raw))
        print(f'concurrency=2: {table[-1]}')

        Path('metrics').mkdir(exist_ok=True)
        Path('metrics/load-raw.jsonl').write_text(
            '\n'.join(json.dumps(r, ensure_ascii=False) for r in raw) + '\n'
        )
        Path('docs').mkdir(exist_ok=True)
        with Path('docs/load-table.csv').open('w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            w.writeheader()
            w.writerows(table)
        print('→ metrics/load-raw.jsonl, docs/load-table.csv')
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


if __name__ == '__main__':
    main()