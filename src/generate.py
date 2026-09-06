"""Генерация ответа на промпт из конфига."""

from config import load_params
from model import generate, load_model, set_seed


def main() -> None:
    params = load_params()
    set_seed(params["generate"]["seed"]) # устанавливаем seed
    tokenizer, model = load_model(params)
    print(f"Модель: {params['model']['name']}")
    text, n_tokens = generate(tokenizer, model, params, params["bench"]["prompt"])

    print(text)
    print(f"\n[{n_tokens} токенов]")


if __name__ == "__main__":
    main()
