import pickle

import pandas as pd

from data.loading import ARTIFACTS_DIR, load_products

MODELS_PATH = ARTIFACTS_DIR / "models.pkl"

_models = {}      # назва моделі -> об'єкт
_products = None  # product_id -> (назва, полиця)


def register(model_name, model):
    _models[model_name] = model


def save_models(names):
    """Зберігає моделі для дашборду. names: {назва в дашборді: назва, під якою модель зареєстрована}."""
    ARTIFACTS_DIR.mkdir(exist_ok=True)
    with open(MODELS_PATH, "wb") as f:
        pickle.dump({new: _models[old] for new, old in names.items()}, f)


def load_models():
    """Завантажує збережені моделі (нічого не тренує). Повертає список їхніх назв."""
    with open(MODELS_PATH, "rb") as f:
        saved = pickle.load(f)
    for name, model in saved.items():
        register(name, model)
    return list(saved)


def recommend(basket, model_name, k=10):
    """Повертає таблицю: position, product_id, product_name, aisle, score."""

    global _products
    if model_name not in _models:
        raise ValueError(f"Unknown model: {model_name}")
    if _products is None:
        products = load_products()
        _products = dict(zip(products["product_id"], zip(products["product_name"], products["aisle"])))

    scores = _models[model_name].recommend(basket, k)
    return pd.DataFrame({
        "position": range(1, len(scores) + 1),
        "product_id": scores.index,
        "product_name": [_products[p][0] for p in scores.index],
        "aisle": [_products[p][1] for p in scores.index],
        "score": scores.values,
    })
