from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] # мабуть краще змінну оточення зробити, але поки так
RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
ARTIFACTS_DIR = ROOT / "artifacts"

BASKETS_PATH = PROCESSED_DIR / "baskets.parquet"
PRODUCTS_PATH = PROCESSED_DIR / "products.parquet"


def load_baskets(split=None):
    """order_id, user_id, product_id, add_to_cart_order, reordered, split"""
    df = pd.read_parquet(BASKETS_PATH)
    if split is not None:
        df = df[df["split"] == split]
    return df


def load_products():
    """product_id, product_name, aisle, department"""
    return pd.read_parquet(PRODUCTS_PATH)
