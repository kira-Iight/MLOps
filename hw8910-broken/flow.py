"""Prefect-flow выпуска модели.

Задачи:
  1. prepare_inputs    — кешируемая подготовка (вход + config → препроцессор).
  2. fetch_artifact    — чтение/проверка манифеста и отчёта (не обучает, читает training run).
  3. evaluate_gate     — validate_release + quality gate (полная проверка отчёта).
  4. publish_release   — идемпотентная атомарная публикация в accepted.json + журнал.

Кеш подготовки привязан к содержимому input и config через release.cache_key.
Сохраняется на диск через persist_result: результат переживает завершение процесса.
"""
from pathlib import Path

from prefect import flow, task
from prefect.cache_policies import INPUTS
from prefect.tasks import task_input_hash

import release
import lab


@task(
    name="prepare_inputs",
    # Кеш-ключ учитывает содержимое input и ВСЕ поля config — тот же
    # контракт, что release.cache_key. При изменении байтов или версии
    # токенизатора/шаблона/max_length/кода — задача выполнится заново.
    cache_key_fn=lambda context, parameters: release.cache_key(
        parameters["data_path"], parameters["config"]
    ),
    persist_result=True,
    cache_expiration=None,
)
def prepare_inputs(data_path: Path, config: dict) -> dict:
    """Подготовка (токенизация/валидация входа). Кешируется по содержимому."""
    data_path = Path(data_path)
    payload = data_path.read_bytes()
    result = {
        "input_sha256": release._sha256(payload),
        "rows": len(payload.splitlines()),
        "config": config,
    }
    # Дублируем в файловый кеш на диске: следующий процесс (или subprocess)
    # прочитает без обращения к Prefect-серверу.
    cache = data_path.parent / "cache"
    cache.mkdir(exist_ok=True)
    key = release.cache_key(data_path, config)
    (cache / (key + ".json")).write_text(
        __import__("json").dumps(result, ensure_ascii=False, indent=2)
    )
    return result


@task(name="fetch_artifact")
def fetch_artifact(workspace: Path, manifest_name: str = "manifest-B.json",
                   report_name: str = "report-B.json") -> tuple[dict, dict]:
    """Чтение манифеста и отчёта из workspace.

    По контракту стадия может читать готовый training run из ДЗ 6+7.
    Здесь читаем STUB-фикстуры. В реальном проекте — артефакт из MLflow.
    """
    workspace = Path(workspace)
    manifest = lab.read(workspace / manifest_name)
    report = lab.read(workspace / report_name)
    return manifest, report


@task(name="evaluate_gate")
def evaluate_gate(manifest: dict, report: dict) -> dict:
    """Полная проверка отчёта. ValueError → flow останавливается до публикации."""
    release.validate_release(manifest, report)
    # В реальном проекте здесь вызов quality_gate ДЗ 6+7 на сохранённых
    # baseline/candidate-eval. Для STUB-фикстуры достаточно validate_release.
    return {
        "manifest": manifest,
        "report": report,
        "release_id": lab._release_id(manifest, report),
    }


@task(name="publish_release")
def publish_release(decision: dict, workspace: Path) -> dict:
    """Идемпотентная публикация: журнал → accepted.json.

    Порядок (журнал первым, потом accepted) важен для восстановления после
    сбоя между шагами: повтор увидит release_id в журнале и завершит публикацию
    без дубликата.
    """
    workspace = Path(workspace)
    manifest = decision["manifest"]
    report = decision["report"]
    rid = decision["release_id"]

    pubs = lab._publications(workspace)
    if any(p.get("release_id") == rid for p in pubs):
        # Идемпотентный повтор: журнал уже содержит событие.
        accepted = lab.read(workspace / "accepted.json")
        if accepted.get("model_id") != manifest["model_id"]:
            # Сбой между журналом и accepted: восстанавливаем.
            release.publish(manifest, report, workspace / "accepted.json")
        return {"status": "repeat", "release_id": rid}

    lab._append_publication(workspace, {
        "release_id": rid,
        "model_id": manifest["model_id"],
        "runtime_sha256": manifest["runtime_sha256"],
        "evaluation_run_id": report["evaluation_run_id"],
    })
    release.publish(manifest, report, workspace / "accepted.json")
    return {"status": "published", "release_id": rid}


@flow(name="release-flow", log_prints=True)
def release_flow(workspace: Path = Path(".lab"),
                 fail_after_prepare: bool = False,
                 report_name: str = "report-B.json") -> dict:
    """Полный flow выпуска: prepare → fetch → gate → publish."""
    workspace = Path(workspace)
    config = lab.read(workspace / "config.json")

    prepared = prepare_inputs(workspace / "input.txt", config)
    print(f"prepared: input_sha256={prepared['input_sha256'][:12]}...")

    if fail_after_prepare:
        raise RuntimeError("Инъекция сбоя после подготовки")

    manifest, report = fetch_artifact(workspace, report_name=report_name)
    decision = evaluate_gate(manifest, report)
    result = publish_release(decision, workspace)
    print(f"result: {result}")
    return result


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=Path(".lab"))
    ap.add_argument("--fail-after-prepare", action="store_true")
    ap.add_argument("--report", default="report-B.json")
    args = ap.parse_args()

    release_flow(
        workspace=args.workspace,
        fail_after_prepare=args.fail_after_prepare,
        report_name=args.report,
    )