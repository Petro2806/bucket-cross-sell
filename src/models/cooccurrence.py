import numpy as np
import pandas as pd

from models.catalog import product_counts
from models.popularity import Popularity


def count_pairs(train, ordered=False):
    """Скільки разів кожну пару товарів (a, b) купили в одному кошику.

    ordered=False: пара враховується незалежно від порядку.
    ordered=True:  лише якщо b додали в кошик ПІСЛЯ a.
    """
    items = train[["order_id", "product_id", "add_to_cart_order"]]
    pairs = items.merge(items, on="order_id", suffixes=("_a", "_b"))
    if ordered:
        pairs = pairs[pairs["add_to_cart_order_a"] < pairs["add_to_cart_order_b"]]
    else:
        pairs = pairs[pairs["product_id_a"] != pairs["product_id_b"]]
    return pairs.groupby(["product_id_a", "product_id_b"]).size().rename("together").reset_index()


class CoOccurrence:
    """Рекомендує товари, які часто купують разом з товарами кошика.

    Нормалізація (n_a - у скількох кошиках є товар a, N - кількість кошиків):
    - "conditional": together / n_a                = P(b | a), частка кошиків з a, де є й b
    - "cosine":      together / sqrt(n_a * n_b)
    - "lift":        together * N / (n_a * n_b)    = у скільки разів частіше разом, ніж випадково
    Score товару = сума схожостей з усіма товарами кошика.
    Якщо сусідів менше ніж k - доповнюємо популярними товарами (зі score 0).
    """

    def __init__(self, train, catalog, similarity="lift", min_support=3, ordered=False):
        train = train[train["product_id"].isin(catalog)]
        n = product_counts(train)
        n_orders = train["order_id"].nunique()

        pairs = count_pairs(train, ordered)
        pairs = pairs[pairs["together"] >= min_support]
        n_a = pairs["product_id_a"].map(n)
        n_b = pairs["product_id_b"].map(n)

        if similarity == "conditional":
            pairs["score"] = pairs["together"] / n_a
        elif similarity == "cosine":
            pairs["score"] = pairs["together"] / np.sqrt(n_a * n_b)
        elif similarity == "lift":
            pairs["score"] = pairs["together"] * n_orders / (n_a * n_b)
        else:
            raise ValueError(f"Unknown similarity: {similarity}")

        self.neighbors = {a: group.set_index("product_id_b")["score"]
                          for a, group in pairs.groupby("product_id_a")}
        self.popularity = Popularity(train, catalog)

    def recommend(self, basket, k=10):
        basket_neighbors = [self.neighbors[p] for p in basket if p in self.neighbors]

        recs = pd.Series(dtype=float)
        if basket_neighbors:
            scores = pd.concat(basket_neighbors).groupby(level=0).sum() 
            recs = scores.drop(basket, errors="ignore").nlargest(k)


        if len(recs) < k:
            fill = self.popularity.recommend(list(basket) + list(recs.index), k - len(recs))
            recs = pd.concat([recs, pd.Series(0.0, index=fill.index)])
        return recs
