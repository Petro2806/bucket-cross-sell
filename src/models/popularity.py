from models.catalog import product_counts


class Popularity:
    def __init__(self, train, catalog):
        counts = product_counts(train)
        self.scores = counts[counts.index.isin(catalog)]

    def recommend(self, basket, k=10):
        return self.scores[~self.scores.index.isin(basket)].head(k)
