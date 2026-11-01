MIN_COUNT = 5


def product_counts(train):
    return train.groupby("product_id")["order_id"].nunique().sort_values(ascending=False)


def build_catalog(train, min_count=MIN_COUNT):
    counts = product_counts(train)
    return set(counts[counts >= min_count].index)
