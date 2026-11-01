import numpy as np
import pandas as pd

from models.catalog import product_counts
from models.cooccurrence import count_pairs
from models.popularity import Popularity


class EASE:
    """Лінійна регресія: кожен товар передбачаємо через усі інші товари кошика (Steck, 2019).

    B[a, b] = наскільки наявність a в кошику підвищує (або знижує, якщо < 0) шанс b,
    з урахуванням усіх інших товарів кошика.
    lam - ridge-регуляризація: штрафує великі ваги, щоб рідкісні товари не давали шуму.
    Score товару = сума ваг від усіх товарів кошика. Не ймовірність, може бути від'ємним.
    """

    def __init__(self, train, catalog, lam=500):
        train = train[train["product_id"].isin(catalog)]
        n = product_counts(train)
        self.products = np.array(sorted(n.index))                   # номер рядка/колонки -> product_id
        self.index = {p: i for i, p in enumerate(self.products)}    # product_id -> номер

        pairs = count_pairs(train)
        G = np.zeros((len(self.products), len(self.products)), dtype=np.float32)
        G[pairs["product_id_a"].map(self.index), pairs["product_id_b"].map(self.index)] = pairs["together"]
        np.fill_diagonal(G, n[self.products].values + lam)          # + lam: ridge-регуляризація

        P = np.linalg.inv(G)
        del G                       # матриця велика (~0.85 ГБ), одразу звільняємо пам'ять
        P /= -np.diag(P)            # те саме, що B = -P / diag(P), але без ще однієї копії матриці
        np.fill_diagonal(P, 0)
        self.B = P

        self.popularity = Popularity(train, catalog)

    def recommend(self, basket, k=10):
        basket_idx = [self.index[p] for p in basket if p in self.index]

        recs = pd.Series(dtype=float)
        if basket_idx:
            scores = pd.Series(self.B[basket_idx].sum(axis=0), index=self.products) 
            scores = scores.drop(basket, errors="ignore")
            recs = scores[scores > 0].nlargest(k)

        if len(recs) < k:
            fill = self.popularity.recommend(list(basket) + list(recs.index), k - len(recs))
            recs = pd.concat([recs, pd.Series(0.0, index=fill.index)])
        return recs
