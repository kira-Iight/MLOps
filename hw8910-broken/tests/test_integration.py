"""Интеграционные тесты: HTTP + subprocess. Stub backend, но реальные
процессы, сокеты, файловая система.

Все тесты должны падать на ИСХОДНОЙ версии прототипа и проходить на
исправленной. Логи до/после сохраняются в logs/.
"""
import hashlib
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def http(method, url, body=None, timeout=3):
    """Обёртка над urllib: возвращает (status, json_body)."""
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


class ServiceProcess:
    """Контекстный менеджер: поднимает service.py в отдельном процессе."""

    def __init__(self, workspace: Path, warmup: float = 2.0):
        self.workspace = workspace
        self.warmup = warmup
        self.port = free_port()
        self.proc = None

    def __enter__(self):
        self.proc = subprocess.Popen(
            [sys.executable, 'service.py',
             '--workspace', str(self.workspace),
             '--port', str(self.port),
             '--warmup-seconds', str(self.warmup)],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        # Ждём, пока порт начнёт принимать соединения
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(('127.0.0.1', self.port), timeout=0.1):
                    return self
            except OSError:
                if self.proc.poll() is not None:
                    out = self.proc.stdout.read() if self.proc.stdout else ''
                    raise RuntimeError(f'service.py упал при старте: {out}')
                time.sleep(0.05)
        raise RuntimeError(f'service.py не открыл порт {self.port} за 10 секунд')

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


def run_lab(workspace: Path, *args, cwd=ROOT, timeout=30):
    """Запуск lab.py с аргументами. Возвращает (returncode, stdout+stderr)."""
    cmd = [sys.executable, 'lab.py', '--workspace', str(workspace), *args]
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    return r.returncode, (r.stdout or '') + (r.stderr or '')


class IntegrationTests(unittest.TestCase):
    """Все тесты используют отдельный временный workspace."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.ws = Path(self.tmp.name) / 'lab'
        rc, out = run_lab(self.ws, 'init')
        assert rc == 0, f'init упал: {out}'

    # =========================================================================
    # HTTP-тесты (3+)
    # =========================================================================

    def test_ready_503_before_warmup(self):
        """Дефект 5: /ready отдаёт 503 до завершения прогрева, 200 после."""
        with ServiceProcess(self.ws, warmup=2.0) as svc:
            time.sleep(0.2)
            code, body = http('GET', svc.url('/ready'))
            self.assertEqual(code, 503, f'до прогрева ждали 503, получили {code}: {body}')
            self.assertFalse(body['ready'])

            time.sleep(2.5)
            code, body = http('GET', svc.url('/ready'))
            self.assertEqual(code, 200, f'после прогрева ждали 200, получили {code}')
            self.assertTrue(body['ready'])

    def test_invalid_requests_4xx_without_backend_call(self):
        """Дефект 7: невалидные запросы — 4xx, calls не растёт."""
        with ServiceProcess(self.ws, warmup=0.5) as svc:
            time.sleep(0.7)
            _, d = http('GET', svc.url('/debug'))
            baseline_calls = d['calls']

            invalid = [
                {'prompt': '', 'max_new_tokens': 5},
                {'prompt': '  ', 'max_new_tokens': 5},
                {'prompt': 'x', 'max_new_tokens': 0},
                {'prompt': 'x', 'max_new_tokens': 100000},
                {'prompt': 'x', 'max_new_tokens': '5'},
                {'prompt': 'x', 'max_new_tokens': True},
            ]
            for body in invalid:
                code, _ = http('POST', svc.url('/generate'), body)
                self.assertIn(code, (400, 422),
                              f'для {body} ждали 4xx, получили {code}')

            _, d = http('GET', svc.url('/debug'))
            self.assertEqual(d['calls'], baseline_calls,
                             'невалидные запросы не должны доходить до backend')

            # Валидный — должен пройти и увеличить calls
            code, body = http('POST', svc.url('/generate'),
                              {'prompt': 'hello', 'max_new_tokens': 4})
            self.assertEqual(code, 200)
            _, d = http('GET', svc.url('/debug'))
            self.assertEqual(d['calls'], baseline_calls + 1)

    def test_loaded_version_not_changed_by_publish(self):
        """Дефект 6: публикация B в другом процессе не меняет model_id
        у запущенного сервиса — до перезапуска."""
        # Запустили сервис с текущим accepted = fixture-A
        with ServiceProcess(self.ws, warmup=0.5) as svc:
            time.sleep(0.7)
            code, body = http('POST', svc.url('/generate'),
                              {'prompt': 'x', 'max_new_tokens': 2})
            self.assertEqual(code, 200)
            self.assertEqual(body['model_id'], 'fixture-A',
                             'свежий стенд должен стартовать с A')

            # В другом процессе публикуем B
            rc, out = run_lab(self.ws, 'run')
            self.assertEqual(rc, 0, f'публикация упала: {out}')
            accepted = json.loads((self.ws / 'accepted.json').read_text())
            self.assertEqual(accepted['model_id'], 'fixture-B')

            # Сервис всё ещё отвечает A — модель в памяти не поменялась
            code, body = http('POST', svc.url('/generate'),
                              {'prompt': 'x', 'max_new_tokens': 2})
            self.assertEqual(code, 200)
            self.assertEqual(body['model_id'], 'fixture-A',
                             'без перезапуска сервис должен остаться на A')

        # После управляемого перезапуска сервис видит B
        with ServiceProcess(self.ws, warmup=0.5) as svc2:
            time.sleep(0.7)
            code, body = http('POST', svc2.url('/generate'),
                              {'prompt': 'x', 'max_new_tokens': 2})
            self.assertEqual(code, 200)
            self.assertEqual(body['model_id'], 'fixture-B',
                             'после перезапуска сервис должен видеть B')

    # =========================================================================
    # Subprocess-тесты (2+)
    # =========================================================================

    def test_cache_invalidated_by_input_bytes(self):
        """Дефект 1: изменение input.txt по тому же пути меняет кеш."""
        rc, out1 = run_lab(self.ws, 'run')
        self.assertEqual(rc, 0, out1)
        self.assertIn('PREPARED', out1)

        # Тот же input — CACHE_HIT
        rc, out2 = run_lab(self.ws, 'run')
        self.assertEqual(rc, 0, out2)
        self.assertIn('CACHE_HIT', out2)

        # Меняем байты — должен быть PREPARED с новым хешем
        (self.ws / 'input.txt').write_text('Другие данные\n')
        rc, out3 = run_lab(self.ws, 'run')
        self.assertEqual(rc, 0, out3)
        self.assertIn('PREPARED', out3,
                      f'после изменения input.txt должен быть PREPARED, а не CACHE_HIT:\n{out3}')

    def test_repeat_publication_idempotent(self):
        """Дефект 4: два одинаковых выпуска → одна запись в журнале."""
        rc1, _ = run_lab(self.ws, 'run')
        rc2, _ = run_lab(self.ws, 'run')
        self.assertEqual(rc1, 0)
        self.assertEqual(rc2, 0)

        pub = (self.ws / 'publications.jsonl').read_text().splitlines()
        self.assertEqual(len(pub), 1,
                         f'ждали 1 запись после двух запусков, получили {len(pub)}')

    def test_incomplete_report_rejected_unchanged_accepted(self):
        """Дефект 2: неполный отчёт не меняет accepted.json."""
        before = (self.ws / 'accepted.json').read_bytes()
        before_sha = hashlib.sha256(before).hexdigest()

        rc, out = run_lab(self.ws, 'run', '--report', 'report-incomplete.json')
        self.assertNotEqual(rc, 0, f'run на неполном отчёте должен упасть:\n{out}')

        after = (self.ws / 'accepted.json').read_bytes()
        after_sha = hashlib.sha256(after).hexdigest()
        self.assertEqual(before_sha, after_sha,
                         'accepted.json должен быть побайтно неизменён')

    def test_failure_after_prepare_uses_cache_on_retry(self):
        """Восстановление: падение после подготовки → повтор берёт кеш."""
        rc, out = run_lab(self.ws, 'run', '--fail-after-prepare')
        self.assertNotEqual(rc, 0, 'сбой должен вернуть ненулевой код')
        self.assertIn('PREPARED', out, 'первый прогон должен был сделать подготовку')

        rc, out = run_lab(self.ws, 'run')
        self.assertEqual(rc, 0)
        self.assertIn('CACHE_HIT', out,
                      f'повтор должен взять из кеша, а не пересчитывать:\n{out}')

    # =========================================================================
    # Mock-тест (атомарность при ошибке диска)
    # =========================================================================

    def test_publish_atomic_on_replace_error(self):
        """Дефект 3: ошибка os.replace → accepted.json побайтно неизменён."""
        sys.path.insert(0, str(ROOT))
        try:
            import importlib
            import release as rel
            importlib.reload(rel)
        finally:
            sys.path.pop(0)

        target = self.ws / 'accepted.json'
        before = target.read_bytes()

        manifest = json.loads((self.ws / 'manifest-B.json').read_text())
        report = json.loads((self.ws / 'report-B.json').read_text())

        with mock.patch('os.replace', side_effect=OSError('disk full')):
            with self.assertRaises(OSError):
                rel.publish(manifest, report, target)

        after = target.read_bytes()
        self.assertEqual(before, after,
                         'при ошибке записи accepted.json должен остаться неизменён')

    # =========================================================================
    # Prefect flow (subprocess) — Phase 4
    # =========================================================================

    def test_flow_publishes_once_and_repeats_idempotently(self):
        """Flow: первый run — published, второй — repeat, журнал = 1."""
        ws = Path(self.tmp.name) / 'flow1'
        rc, out = run_lab(ws, 'init')
        self.assertEqual(rc, 0, out)

        # Prefect пишет всё в stderr — используем stdout+stderr
        r1 = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        c1 = r1.stdout + r1.stderr
        self.assertEqual(r1.returncode, 0, c1[-1500:])
        self.assertIn("'status': 'published'", c1, f'нет published в логе:\n{c1[-1000:]}')

        r2 = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        c2 = r2.stdout + r2.stderr
        self.assertEqual(r2.returncode, 0, c2[-1500:])
        self.assertIn("'status': 'repeat'", c2, f'нет repeat в логе:\n{c2[-1000:]}')

        pubs = (ws / 'publications.jsonl').read_text().splitlines()
        self.assertEqual(len(pubs), 1, f'ожидали 1 публикацию, got {len(pubs)}')

    def test_flow_cache_invalidated_on_input_change(self):
        """Flow: смена input.txt → prepare даёт новый input_sha256."""
        import re

        ws = Path(self.tmp.name) / 'flow2'
        rc, _ = run_lab(ws, 'init')
        self.assertEqual(rc, 0)

        r1 = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        c1 = r1.stdout + r1.stderr
        self.assertEqual(r1.returncode, 0, c1[-1500:])
        m1 = re.search(r'input_sha256=([0-9a-f]+)', c1)
        self.assertIsNotNone(m1, f'не нашли input_sha256 в логе:\n{c1[-1000:]}')

        # Меняем байты — prepare должен выполниться заново
        (ws / 'input.txt').write_text('Совсем другие данные\n')
        r2 = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        c2 = r2.stdout + r2.stderr
        self.assertEqual(r2.returncode, 0, c2[-1500:])
        m2 = re.search(r'input_sha256=([0-9a-f]+)', c2)
        self.assertIsNotNone(m2, f'не нашли input_sha256 в логе:\n{c2[-1000:]}')

        self.assertNotEqual(
            m1.group(1), m2.group(1),
            f'prepared должен дать новый хеш: {m1.group(1)} vs {m2.group(1)}',
        )

    def test_flow_failure_after_prepare_recovers_from_cache(self):
        """Flow: сбой после prepare → повтор берёт prepare из кеша."""
        ws = Path(self.tmp.name) / 'flow3'
        rc, _ = run_lab(ws, 'init')
        self.assertEqual(rc, 0)

        r_fail = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws), '--fail-after-prepare'],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertNotEqual(r_fail.returncode, 0, 'сбой должен вернуть ненулевой код')

        r_retry = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        c_retry = r_retry.stdout + r_retry.stderr
        self.assertEqual(r_retry.returncode, 0, c_retry[-1500:])
        # После успешного prepare (в первом прогоне) повтор должен взять его из кеша
        self.assertIn('Cached', c_retry,
                      f'повтор должен взять prepare из кеша:\n{c_retry[-1000:]}')

    def test_flow_cache_invalidated_on_config_change(self):
        """Flow: смена config.json → prepare_inputs выполняется заново.

        Проверяем через НАШ file cache: prepare пишет cache/<key>.json.
        Удаляем cache перед вторым запуском и смотрим, появился ли новый файл.
        Если бы Prefect считал задачу Cached — файл не появился бы (тело не исполнилось).
        Уникальный input изолирует тест от Prefect-кеша других тестов.
        """
        import json
        import shutil
        import uuid

        ws = Path(self.tmp.name) / 'flow4'
        rc, _ = run_lab(ws, 'init')
        self.assertEqual(rc, 0)

        # Уникальный input — изоляция от Prefect-кеша предыдущих тестов
        (ws / 'input.txt').write_text(f'unique-cfg-{uuid.uuid4()}\n')

        r1 = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(r1.returncode, 0, (r1.stdout + r1.stderr)[-1500:])

        cache_dir = ws / 'cache'
        self.assertEqual(len(list(cache_dir.glob('*.json'))), 1,
                         'после первого flow в cache/ должен быть ровно 1 файл')

        # Удаляем file cache: если задача исполнится, файл появится снова
        shutil.rmtree(cache_dir)

        # Меняем max_length → cache key должен измениться
        p = ws / 'config.json'
        cfg = json.loads(p.read_text())
        cfg['max_length'] = 256
        p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2))

        r2 = subprocess.run(
            [sys.executable, 'flow.py', '--workspace', str(ws)],
            cwd=ROOT, capture_output=True, text=True, timeout=90,
        )
        self.assertEqual(r2.returncode, 0, (r2.stdout + r2.stderr)[-1500:])

        # Если задача исполнилась — prepare снова записал cache/<new_key>.json
        files_after = list(cache_dir.glob('*.json'))
        self.assertEqual(
            len(files_after), 1,
            f'prepare должен исполниться после смены config (cache был удалён), '
            f'но file cache пуст — значит Prefect считает задачу Cached. '
            f'stdout+stderr:\n{(r2.stdout + r2.stderr)[-1500:]}',
        )


if __name__ == '__main__':
    unittest.main()