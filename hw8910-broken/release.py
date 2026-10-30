"""Исправленная реализация выпуска. Контракт: CONTRACTS.txt."""
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path


def _canonical(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'),
        ensure_ascii=False, allow_nan=False,
    ).encode('utf-8')


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _is_sha256(value) -> bool:
    return (
        isinstance(value, str) and len(value) == 64
        and all(c in '0123456789abcdef' for c in value)
    )


def cache_key(data_path: Path, config: dict) -> str:
    """SHA-256 от байтов файла + канонического config.

    Не зависит от пути — одинаковые байты в разных путях дают один ключ.
    Изменение любого поля config (ревизия токенизатора, шаблон,
    max_length, версия кода) меняет ключ.
    """
    data_bytes = Path(data_path).read_bytes()
    payload = {
        'data_sha256': _sha256(data_bytes),
        'config': config,
    }
    return _sha256(_canonical(payload))


_REQUIRED_MANIFEST = (
    'schema_version', 'model_id', 'artifact_uri',
    'model_sha256', 'protocol_sha256', 'golden_sha256',
    'runtime_sha256', 'runtime',
)
_REQUIRED_REPORT = (
    'status', 'passed', 'evaluation_run_id',
    'model_id', 'model_sha256', 'protocol_sha256',
    'golden_sha256', 'runtime_sha256', 'metrics',
)
_HASH_FIELDS = (
    'model_sha256', 'protocol_sha256', 'golden_sha256', 'runtime_sha256',
)


def validate_release(manifest: dict, report: dict) -> None:
    """Полная проверка манифеста и отчёта. ValueError при любом нарушении."""
    if not isinstance(manifest, dict):
        raise ValueError('manifest: не словарь')
    if not isinstance(report, dict):
        raise ValueError('report: не словарь')

    for field in _REQUIRED_MANIFEST:
        if field not in manifest:
            raise ValueError(f'manifest: нет поля {field}')
    for field in _REQUIRED_REPORT:
        if field not in report:
            raise ValueError(f'report: нет поля {field}')

    # schema_version — именно int 1, не True, не "1", не 2
    sv = manifest['schema_version']
    if not isinstance(sv, int) or isinstance(sv, bool) or sv != 1:
        raise ValueError(f'manifest.schema_version: ожидалось int 1, есть {sv!r}')

    for field in ('model_id', 'artifact_uri'):
        v = manifest[field]
        if not isinstance(v, str) or not v.strip():
            raise ValueError(f'manifest.{field}: пустая или невалидная строка')

    for field in _HASH_FIELDS:
        if not _is_sha256(manifest[field]):
            raise ValueError(f'manifest.{field}: не SHA-256 (64 hex lowercase)')

    runtime = manifest['runtime']
    if not isinstance(runtime, dict) or not runtime:
        raise ValueError('manifest.runtime: пустой или не словарь')
    expected = _sha256(_canonical(runtime))
    if manifest['runtime_sha256'] != expected:
        raise ValueError(
            f'manifest.runtime_sha256: не совпадает с пересчётом ({expected})'
        )

    if report['status'] != 'complete':
        raise ValueError(f'report.status: {report["status"]!r} != "complete"')

    # passed is True — именно bool, не строка "true" и не 1
    if report['passed'] is not True:
        raise ValueError(f'report.passed: {report["passed"]!r} — нужен True (bool)')

    if not isinstance(report['evaluation_run_id'], str) or not report['evaluation_run_id'].strip():
        raise ValueError('report.evaluation_run_id: пустая строка')

    if report['model_id'] != manifest['model_id']:
        raise ValueError('report.model_id != manifest.model_id')
    for field in _HASH_FIELDS:
        if report[field] != manifest[field]:
            raise ValueError(f'report.{field} != manifest.{field}')

    metrics = report['metrics']
    if not isinstance(metrics, dict):
        raise ValueError('report.metrics: не словарь')
    for field in ('accuracy', 'format_rate'):
        if field not in metrics:
            raise ValueError(f'report.metrics.{field}: нет поля')
        v = metrics[field]
        if isinstance(v, bool):
            raise ValueError(f'report.metrics.{field}: bool запрещён')
        if not isinstance(v, (int, float)):
            raise ValueError(f'report.metrics.{field}: не число')
        if not math.isfinite(v):
            raise ValueError(f'report.metrics.{field}: NaN/Inf')
        if not (0 <= v <= 1):
            raise ValueError(f'report.metrics.{field}: вне [0, 1]')


def publish(manifest: dict, report: dict, target: Path) -> None:
    """Атомарная публикация: validate → temp → fsync → os.replace.

    При отказе validate или ошибке записи целевой файл побайтно не тронут.
    Повторная публикация того же manifest даёт те же байты.
    """
    validate_release(manifest, report)

    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)

    payload = json.dumps(manifest, ensure_ascii=False, indent=2).encode('utf-8')

    fd, tmp_path = tempfile.mkstemp(
        dir=str(target.parent), prefix=f'.{target.name}.', suffix='.tmp',
    )
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, target)
        try:
            dir_fd = os.open(str(target.parent), os.O_RDONLY)
            try:
                os.fsync(dir_fd)
            finally:
                os.close(dir_fd)
        except OSError:
            pass
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise