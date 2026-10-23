# Разбор четырёх дефектов части A

## Дефект 1: Все кандидаты в одном run

**В чём был.** В `src/sweep.py` был один вложенный run `all-candidates`, в который попадали все 6 моделей. `mlflow.search_runs` по `parentRunId` находил только 1 run вместо 6.

**Как проявлялся.** `check.py` проверка 1 падала: «отдельных кандидатов: 1». `make candidate` не мог выбрать лучшего — сравнивать было нечего. MLflow UI показывал один run с последними метриками вместо кривых шести моделей.

**Как исправлен.** Убрал внешний `with mlflow.start_run("all-candidates")`, обернул каждый `run_candidate()` в отдельный `with mlflow.start_run(candidate["name"], nested=True)`.

**Числом.** До: 1 run. После: 6 runs, у каждого свой `run_id`, свои `curve_train`/`curve_val` по 4 шага. `make report` выдаёт таблицу из 6 строк (`docs/comparison.csv`).

## Дефект 2: Артефакты теряются при пересоздании контейнера

**В чём был.** В `compose.yaml` volume `artifacts` объявлен в секции `volumes:`, но не смонтирован в сервис `mlflow`. Артефакты (`/mlartifacts`) жили в writable-слое контейнера.

**Как проявлялся.** `make check-storage` пересоздаёт контейнер через `docker compose up --force-recreate mlflow`. Все артефакты (модели, samples.json) исчезали. `check.py` проверка 8 не могла скачать `persistence/probe.txt` — run падал с 404. Все runs после `--force-recreate` становились «модель без артефактов».

**Как исправлен.** Добавил `volumes: - artifacts:/mlartifacts` в сервис `mlflow` в `compose.yaml`.

**Числом.** До: `check-storage` — `FAIL: артефакт потерян`. После: `PASS: новый контейнер вернул тот же артефакт, SHA256 совпал`.

## Дефект 3: Хеши train/val не логируются

**В чём был.** В `src/sweep.py`: `versions = {k: v for k, v in provenance(p).items() if not k.startswith("data_")}` — специально вырезались `data_train_sha256` и `data_val_sha256`.

**Как проявлялся.** `check.py` проверка 2 падала: «Run <id>: нет актуального data_train_sha256». Sweep с изменённым train.pt переиспользовал старые runs — provenance не сверялся. Сравнение моделей было нечестным.

**Как исправлен.** Убрал фильтр `if not k.startswith("data_")`. Теперь `provenance(p)` целиком идёт в теги parent и child runs.

**Числом.** До: 4 тега из 6. После: 6 тегов — `data_train_sha256=0cdcdcce...`, `data_val_sha256=98ab6fc9...`. Проверка 2 зелёная.

## Дефект 4: Регистрация в реестр без проверки порога

**В чём был.** В `src/promote.py` функция `promote()` вычисляла `ratio = final_val_loss / base_val_loss`, ставила тег `validation: passed`, но **не вызывала** `gate(run, threshold)`. Кандидат с проваленным порогом всё равно регистрировался.

**Как проявлялся.** `check.py` проверка 6 падала: `promote(client, best, name, "test", 0.0)` с нулевым порогом (гарантированный fail) не поднимал `ValueError`, а регистрировал модель. Сломанный кандидат мог попасть в `candidate` alias, а оттуда — в `champion`.

**Как исправлен.** В `promote()` добавил `ratio = gate(run, threshold)` **до** `mlflow.register_model()`. `gate()` проверяет `run.info.status == "FINISHED"`, `math.isfinite(final_val_loss)`, `math.isfinite(base_val_loss)`, `base_val_loss > 0` и `final_val_loss / base_val_loss <= max_ratio`.

**Числом.** До: `promote(..., threshold=0.0)` регистрировал модель без ошибок. После: тот же вызов поднимает `ValueError("Порог не пройден: val/base=0.0240, максимум 0.0")`. Проверка 6 зелёная.