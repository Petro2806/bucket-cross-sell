import pandas as pd

from data.loading import load_baskets, load_products
from models.catalog import build_catalog
from models.popularity import Popularity

_models = {}           # назва моделі -> об'єкт
_product_names = None  # product_id -> назва 


def register(model_name, model):
    _models[model_name] = model


def load_models():
    train = load_baskets("train")
    catalog = build_catalog(train)
    register("popularity", Popularity(train, catalog))


def recommend(basket, model_name, k=10):
    """Повертає таблицю: position, product_id, product_name, score."""
    
    global _product_names
    if model_name not in _models:
        raise ValueError(f"Unknown model: {model_name}")
    if _product_names is None:
        products = load_products()
        _product_names = dict(zip(products["product_id"], products["product_name"]))

    scores = _models[model_name].recommend(basket, k)
    return pd.DataFrame({
        "position": range(1, len(scores) + 1),
        "product_id": scores.index,
        "product_name": [_product_names[p] for p in scores.index],
        "score": scores.values,
    })
