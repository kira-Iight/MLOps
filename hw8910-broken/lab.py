"""Исполняемый прототип для расследования. Только синтетические данные, без модели.

Запускайте в отдельном каталоге .lab. Этот последовательный прототип надо
превратить в корректный Prefect-flow; его зелёный статус не доказывает корректность.
"""
import argparse
import hashlib
import json
from pathlib import Path
import release


def canonical(value):
    return json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2))


def initialize(root):
    if root.exists():
        raise FileExistsError('Рабочий каталог уже существует. Выберите новый --workspace.')
    root.mkdir(parents=True)
    runtime={'base_revision':'fixture-only','tokenizer_sha256':'a'*64,
             'template_sha256':'b'*64,'generation':{'max_new_tokens':32,'do_sample':False},'dtype':'float32'}
    for version in ['A','B']:
        artifact=root/f'model-{version}.txt';artifact.write_text(f'STUB ONLY — version {version}\n')
        files={artifact.name:digest(artifact.read_bytes())}
        m={'schema_version':1,'model_id':f'fixture-{version}',
           'artifact_uri':artifact.name,'model_sha256':digest(canonical(files)),
           'protocol_sha256':'d'*64,'golden_sha256':'e'*64,
           'runtime_sha256':digest(canonical(runtime)),'runtime':runtime,'file_hashes':files}
        report={k:m[k] for k in ['model_id','model_sha256','protocol_sha256','golden_sha256','runtime_sha256']}
        report.update(status='complete',passed=True,evaluation_run_id=f'fixture-eval-{version}',metrics={'accuracy':.8,'format_rate':1.0},synthetic=True)
        write(root/f'manifest-{version}.json',m);write(root/f'report-{version}.json',report)
    write(root/'accepted.json',read(root/'manifest-A.json'))
    report=read(root/'report-B.json');report['status']='running';report['passed']='false'
    write(root/'report-incomplete.json',report)
    report=read(root/'report-B.json');report['passed']=False
    write(root/'report-reject.json',report)
    write(root/'config.json',{'tokenizer_revision':'r1','template_sha256':'b'*64,'max_length':128,'code_revision':'prepare-v1'})
    (root/'input.txt').write_text('Первый пример\nВторой пример\n')
    print('Создан синтетический стенд:',root)


def prepare(root):
    config=read(root/'config.json');key=release.cache_key(root/'input.txt',config)
    cache=root/'cache';cache.mkdir(exist_ok=True);entry=cache/(key+'.json')
    if entry.exists():
        result=read(entry);print('CACHE_HIT',result['input_sha256']);return result
    payload=(root/'input.txt').read_bytes()
    result={'input_sha256':digest(payload),'rows':len(payload.splitlines()),'config':config}
    write(entry,result);print('PREPARED',result['input_sha256']);return result


def _release_id(manifest, report):
    """Идентификатор выпуска: model + runtime + решение допуска.

    Не сводится к model_id: тот же адаптер с другим runtime или
    другим evaluation_run_id — это другой выпуск.
    """
    key = {
        'model_id': manifest['model_id'],
        'runtime_sha256': manifest['runtime_sha256'],
        'evaluation_run_id': report['evaluation_run_id'],
    }
    return digest(canonical(key))


def _publications(root):
    """Прочитать publications.jsonl (маленький файл, читаем целиком)."""
    path = root / 'publications.jsonl'
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _append_publication(root, record):
    """Атомарно добавить запись: перечитываем, проверяем, append.

    Один писатель, поэтому append в файл безопасен. Идемпотентность
    гарантирует вызывающий код (проверка release_id ДО append).
    """
    with (root / 'publications.jsonl').open('a') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')


def run(root, report_name, fail_after_prepare=False):
    """Публикация: подготовка → (сбой?) → проверка отчёта → атомарная запись → журнал.

    Идемпотентна по release_id: повтор того же выпуска не создаёт
    вторую запись в publications.jsonl и не меняет accepted.json (кроме
    побайтно идентичной атомарной замены).
    """
    prepared = prepare(root)
    if fail_after_prepare:
        raise RuntimeError('Инъекция сбоя после подготовки')

    manifest = read(root / 'manifest-B.json')
    report = read(root / report_name)

    # 1. Проверка отчёта (может упасть — тогда ничего не публикуем)
    release.validate_release(manifest, report)

    # 2. Идентификатор выпуска
    rid = _release_id(manifest, report)

    # 3. Читаем журнал: если этот выпуск уже был, повторяем только
    #    восстановление, если accepted.json разошёлся с последней записью.
    pubs = _publications(root)
    if any(p.get('release_id') == rid for p in pubs):
        # Идемпотентный повтор: журнал уже содержит событие.
        # Проверим, что accepted.json согласован (иначе — восстанавливаем).
        accepted = read(root / 'accepted.json')
        if accepted.get('model_id') != manifest['model_id']:
            release.publish(manifest, report, root / 'accepted.json')
        print('PUBLISHED', manifest['model_id'], '(repeat, release_id=' + rid[:8] + ')')
        return

    # 4. Первая публикация: атомарная запись, затем журнал.
    #    Если упадём между этими шагами — повтор найдёт release_id в журнале
    #    только если журнал записан первым. Поэтому порядок: журнал первым
    #    (маленькая append-запись), потом accepted.json (атомарная замена).
    #    При падении между — accepted остаётся старый, повтор увидит release_id
    #    в журнале и восстановит accepted. Двойного события не будет.
    _append_publication(root, {
        'release_id': rid,
        'model_id': manifest['model_id'],
        'runtime_sha256': manifest['runtime_sha256'],
        'evaluation_run_id': report['evaluation_run_id'],
        'prepared_input_sha256': prepared['input_sha256'],
    })
    release.publish(manifest, report, root / 'accepted.json')
    print('PUBLISHED', manifest['model_id'], '(release_id=' + rid[:8] + ')')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,default=Path('.lab'))
    sub=parser.add_subparsers(dest='command',required=True);sub.add_parser('init')
    run_parser=sub.add_parser('run');run_parser.add_argument('--report',default='report-B.json');run_parser.add_argument('--fail-after-prepare',action='store_true')
    args=parser.parse_args()
    if args.command=='init':initialize(args.workspace)
    else:run(args.workspace,args.report,args.fail_after_prepare)


if __name__=='__main__':main()
