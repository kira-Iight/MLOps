"""FastAPI-версия сервиса. Контракт HTTP тот же, что у service.py (stdlib),
backend внедряется: StubBackend для тестов или ModelBackend для реальной модели.

Прогрев идёт в фоновом потоке — accept начинается сразу, /health и /debug
доступны немедленно, /ready отдаёт 503 до завершения прогрева.
"""
import argparse
import json
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

MAX_NEW_TOKENS = 32
MAX_BODY_BYTES = 65536


class StubBackend:
    def __init__(self, manifest: dict):
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0

    def generate(self, prompt: str, limit: int) -> dict:
        self.calls += 1
        time.sleep(0.01)
        return {'answer': f'STUB ONLY: {self.loaded_id}', 'output_tokens': 1}

    def warmup(self, seconds: float) -> None:
        time.sleep(seconds)


class ModelBackend:
    def __init__(self, manifest: dict, workspace: Path):
        self.manifest = manifest
        self.loaded_id = manifest['model_id']
        self.runtime_sha256 = manifest['runtime_sha256']
        self.calls = 0
        self._model = None
        self._tokenizer = None
        self._torch = None
        self._device = None
        self._workspace = workspace

    def _ensure_loaded(self):
        if self._model is not None:
            return

        artifact_uri = self.manifest['artifact_uri']
        needs_adapter = artifact_uri and artifact_uri != 'base'

        # peft нужен ТОЛЬКО для адаптера. Для baseline (artifact_uri='base')
        # достаточно transformers+torch — не тащим лишнюю зависимость.
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            if needs_adapter:
                from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError(
                f'backend=model требует transformers/torch'
                f'{" и peft" if needs_adapter else ""}. Установите: '
                f'python3 -m pip install torch transformers'
                f'{" peft" if needs_adapter else ""}'
            ) from exc
        rt = self.manifest['runtime']
        base_rev = rt['base_revision']   # 'Qwen/Qwen3-0.6B@sha' или 'Qwen/Qwen3-0.6B'
        artifact_uri = self.manifest['artifact_uri']

        # HF-конвенция: 'repo@revision'. Разбираем на repo + revision,
        # чтобы transformers получил их отдельными аргументами.
        if '@' in base_rev:
            base_name, revision = base_rev.split('@', 1)
        else:
            base_name, revision = base_rev, None

        load_kwargs = {}
        if revision and revision != 'main':
            load_kwargs['revision'] = revision

        self._tokenizer = AutoTokenizer.from_pretrained(base_name, **load_kwargs)
        if self._tokenizer.pad_token_id is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            base_name,
            torch_dtype=getattr(torch, rt.get('dtype', 'float32')),
            **load_kwargs,
        )
        # artifact_uri='base' — маркер baseline без адаптера.
        # Любая другая непустая строка — путь/URI адаптера.
        if artifact_uri and artifact_uri != 'base':
            self._model = PeftModel.from_pretrained(base_model, artifact_uri).eval()
        else:
            self._model = base_model.eval()
        self._device = ('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')
        self._device = ('cuda' if torch.cuda.is_available()
                        else 'mps' if torch.backends.mps.is_available()
                        else 'cpu')
        self._model = self._model.to(self._device)
        self._torch = torch
        self._torch = torch

    def generate(self, prompt: str, limit: int) -> dict:
        self._ensure_loaded()
        torch = self._torch
        gen = self.manifest['runtime']['generation']
        messages = [
            {'role': 'system', 'content': gen.get('system', '')},
            {'role': 'user', 'content': prompt},
        ]
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
            enable_thinking=gen.get('enable_thinking', False),
        )
        batch = self._tokenizer(text, return_tensors='pt').to(self._device)
        self.calls += 1
        with torch.inference_mode():
            out = self._model.generate(
                **batch,
                max_new_tokens=min(limit, gen.get('max_new_tokens', limit)),
                do_sample=gen.get('do_sample', False),
                pad_token_id=self._tokenizer.pad_token_id,
            )
        new_tokens = out[0, batch['input_ids'].shape[1]:]
        answer = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {'answer': answer, 'output_tokens': int(new_tokens.shape[0])}

    def warmup(self, seconds: float) -> None:
        self._ensure_loaded()
        torch = self._torch
        batch = self._tokenizer('warmup', return_tensors='pt').to(self._device)
        with torch.inference_mode():
            self._model.generate(**batch, max_new_tokens=1,
                                 pad_token_id=self._tokenizer.pad_token_id)
        time.sleep(max(0.0, seconds - 0.1))


class ServiceState:
    """Прогрев в фоновом потоке: accept начинается сразу, /ready = 503."""

    def __init__(self, workspace: Path, backend_name: str, warmup_seconds: float):
        self.workspace = workspace
        self.ready = False

        manifest = json.loads((workspace / 'accepted.json').read_text())
        if backend_name == 'stub':
            self.backend = StubBackend(manifest)
        elif backend_name == 'model':
            self.backend = ModelBackend(manifest, workspace)
        else:
            raise ValueError(f'unknown backend {backend_name!r}')

        self.warmup_finished = threading.Event()

        def _warmup():
            try:
                self.backend.warmup(warmup_seconds)
            finally:
                self.warmup_finished.set()
                self.ready = True

        # Фоновый поток, accept НЕ блокируется
        threading.Thread(target=_warmup, daemon=True).start()

    def _validate_request(self, body: dict) -> tuple[str, int]:
        if not isinstance(body, dict):
            raise HTTPException(400, 'body: ожидался JSON-объект')
        prompt = body.get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(400, 'prompt: непустая строка обязательна')
        limit = body.get('max_new_tokens', MAX_NEW_TOKENS)
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise HTTPException(400, 'max_new_tokens: целое число')
        if not (1 <= limit <= MAX_NEW_TOKENS):
            raise HTTPException(400, f'max_new_tokens: 1..{MAX_NEW_TOKENS}')
        return prompt, limit

    def generate(self, body: dict) -> dict:
        prompt, limit = self._validate_request(body)
        started = time.perf_counter()
        result = self.backend.generate(prompt, limit)
        result.update(
            model_id=self.backend.loaded_id,
            runtime_sha256=self.backend.runtime_sha256,
            request_id=str(uuid.uuid4()),
            generation_seconds=time.perf_counter() - started,
        )
        return result


def build_app(workspace: Path, backend_name: str, warmup_seconds: float) -> FastAPI:
    # Конструктор запускает прогрев в фоне и возвращается мгновенно
    state = ServiceState(workspace, backend_name, warmup_seconds)
    app = FastAPI()

    @app.get('/health')
    def health():
        return {'alive': True, 'backend': backend_name.upper()}

    @app.get('/ready')
    def ready():
        if not state.ready:
            return JSONResponse({'ready': False}, status_code=503)
        return {'ready': True}

    @app.get('/debug')
    def debug():
        return {
            'loaded_id': state.backend.loaded_id,
            'calls': state.backend.calls,
            'warmup_finished': state.warmup_finished.is_set(),
        }

    @app.post('/generate')
    async def generate(request: Request):
        content_length = request.headers.get('content-length')
        if content_length and int(content_length) > MAX_BODY_BYTES:
            raise HTTPException(413, 'body too large')
        try:
            body = await request.json()
        except Exception:
            raise HTTPException(400, 'body: невалидный JSON')
        if not state.ready:
            raise HTTPException(503, 'not ready')
        return state.generate(body)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--workspace', type=Path, default=Path('.lab'))
    ap.add_argument('--port', type=int, default=8092)
    ap.add_argument('--warmup-seconds', type=float, default=5.0)
    ap.add_argument('--backend', choices=('stub', 'model'), default='stub')
    args = ap.parse_args()

    import uvicorn
    app = build_app(args.workspace, args.backend, args.warmup_seconds)
    print(f'{args.backend.upper()} backend: http://127.0.0.1:{args.port}', flush=True)
    uvicorn.run(app, host='127.0.0.1', port=args.port, log_level='warning')


if __name__ == '__main__':
    main()