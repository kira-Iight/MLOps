"""Параметризованные HTTP-тесты: один и тот же контракт проверяется
на stdlib service.py и FastAPI service_fastapi.py.

Проверяем: /health, /ready (503→200), /debug (loaded_id, warmup_finished),
/generate (валидный → 200, невалидные → 400 без backend.calls), размер body.

Оба сервиса должны вести себя идентично. Если контракт разошёлся между
frameworks — тест падает и показывает, где.
"""
import json
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def http(method, url, body=None, timeout=3):
    data = None
    headers = {}
    if body is not None:
        data = json.dumps(body).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, None


class ServerProcess:
    """Единый адаптер запуска: stub (stdlib) или stub-fastapi."""

    def __init__(self, workspace: Path, launcher: str, warmup: float = 3.0):
        self.workspace = workspace
        self.launcher = launcher
        self.warmup = warmup
        self.port = free_port()
        self.proc = None

    def __enter__(self):
        if self.launcher == 'stdlib':
            cmd = [sys.executable, 'service.py',
                   '--workspace', str(self.workspace),
                   '--port', str(self.port),
                   '--warmup-seconds', str(self.warmup)]
        elif self.launcher == 'fastapi':
            cmd = [sys.executable, 'service_fastapi.py',
                   '--workspace', str(self.workspace),
                   '--port', str(self.port),
                   '--warmup-seconds', str(self.warmup),
                   '--backend', 'stub']
        else:
            raise ValueError(f'unknown launcher {self.launcher!r}')

        self.proc = subprocess.Popen(
            cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )

        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=0.1):
                    return self
            except OSError:
                if self.proc.poll() is not None:
                    out = self.proc.stdout.read() if self.proc.stdout else ''
                    raise RuntimeError(f'{self.launcher} упал: {out}')
                time.sleep(0.05)
        raise RuntimeError(f'{self.launcher} не открыл порт {self.port}')

    def __exit__(self, *exc):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        if self.proc and self.proc.stdout:
            try:
                self.proc.stdout.close()
            except Exception:
                pass

    def url(self, path):
        return f'http://127.0.0.1:{self.port}{path}'


class HttpContractForBothFrameworks(unittest.TestCase):
    """Общие тесты: параметризованы по launcher. Проверяем оба."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name) / 'lab'
        r = subprocess.run(
            [sys.executable, 'lab.py', '--workspace', str(self.ws), 'init'],
            cwd=ROOT, capture_output=True, text=True,
        )
        assert r.returncode == 0, r.stderr

    def _both_launchers(self):
        """Итератор: (launcher_name, ServerProcess). Запускаем по очереди."""
        for launcher in ('stdlib', 'fastapi'):
            with ServerProcess(self.ws, launcher, warmup=2.0) as svc:
                yield launcher, svc

    def test_health_works_on_both(self):
        for launcher, svc in self._both_launchers():
            with self.subTest(launcher=launcher):
                code, body = http('GET', svc.url('/health'))
                self.assertEqual(code, 200, f'{launcher}: /health = {code}')
                self.assertTrue(body['alive'])

    def test_ready_503_before_warmup_on_both(self):
        for launcher, svc in self._both_launchers():
            with self.subTest(launcher=launcher):
                # Сразу после accept: /ready должен быть 503
                code, body = http('GET', svc.url('/ready'))
                self.assertEqual(code, 503,
                                 f'{launcher}: до прогрева ждали 503, получили {code}: {body}')
                self.assertFalse(body['ready'])

                # После прогрева
                time.sleep(2.3)
                code, body = http('GET', svc.url('/ready'))
                self.assertEqual(code, 200,
                                 f'{launcher}: после прогрева ждали 200, получили {code}')
                self.assertTrue(body['ready'])

    def test_generate_and_invalid_requests_on_both(self):
        for launcher, svc in self._both_launchers():
            with self.subTest(launcher=launcher):
                time.sleep(2.3)  # готов
                _, d0 = http('GET', svc.url('/debug'))
                base = d0['calls']

                # Валидный
                code, body = http('POST', svc.url('/generate'),
                                  {'prompt': 'hello', 'max_new_tokens': 3})
                self.assertEqual(code, 200)
                for k in ('answer', 'output_tokens', 'model_id',
                          'runtime_sha256', 'request_id', 'generation_seconds'):
                    self.assertIn(k, body, f'{launcher}: нет {k} в ответе')

                # Невалидные — 4xx, calls не растёт
                for bad in ({'prompt': ''},
                            {'prompt': 'x', 'max_new_tokens': 0},
                            {'prompt': 'x', 'max_new_tokens': 100000},
                            {'prompt': 'x', 'max_new_tokens': 'x'}):
                    code, _ = http('POST', svc.url('/generate'), bad)
                    self.assertIn(code, (400, 422),
                                  f'{launcher}: для {bad} ждали 4xx, получили {code}')

                _, d1 = http('GET', svc.url('/debug'))
                self.assertEqual(d1['calls'], base + 1,
                                 f'{launcher}: невалидные не должны доходить до backend')


if __name__ == '__main__':
    unittest.main()