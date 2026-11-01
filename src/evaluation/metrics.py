import numpy as np
import pandas as pd

from models.recommend import recommend


def split_baskets(baskets):
    rows = []
    for order_id, items in baskets.sort_values("add_to_cart_order").groupby("order_id")["product_id"]:
        items = items.tolist()
        if len(items) < 2:
            continue
        half = len(items) // 2
        rows.append({"order_id": order_id, "visible": items[:half], "hidden": items[half:]})
    return pd.DataFrame(rows)


def recall_at_k(recommendations, hidden, k=10):
    hits = len(set(recommendations[:k]) & set(hidden))
    return hits / len(hidden)


def ndcg_at_k(recommendations, hidden, k=10):
    hidden = set(hidden)
    dcg = sum(1 / np.log2(pos + 2) for pos, item in enumerate(recommendations[:k]) if item in hidden)
    ideal = sum(1 / np.log2(pos + 2) for pos in range(min(len(hidden), k)))
    return dcg / ideal


def evaluate(model_name, splits, k=10):
    recalls, ndcgs = [], []
    for visible, hidden in zip(splits["visible"], splits["hidden"]):
        recommendations = recommend(visible, model_name, k)["product_id"].tolist()
        recalls.append(recall_at_k(recommendations, hidden, k))
        ndcgs.append(ndcg_at_k(recommendations, hidden, k))

    return {
        f"recall@{k}": np.mean(recalls),
        f"ndcg@{k}": np.mean(ndcgs),
        "n_baskets": len(splits),
    }
