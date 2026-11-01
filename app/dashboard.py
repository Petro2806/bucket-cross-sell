import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from data.loading import ARTIFACTS_DIR, load_baskets, load_products
from models.catalog import MIN_COUNT, product_counts
from models.recommend import load_models, recommend

st.set_page_config(page_title="Basket cross-sell", layout="wide")


@st.cache_resource
def get_models():
    return load_models()


@st.cache_data
def get_products():
    names = load_products().set_index("product_id")["product_name"]
    counts = product_counts(load_baskets("train"))
    catalog = counts[counts >= MIN_COUNT].index
    return names[catalog].to_dict()


models = get_models()
names = get_products()

COLORS = {"new": "rgba(0, 200, 0, 0.3)", "moved": "rgba(255, 200, 0, 0.3)",
          "removed": "rgba(255, 0, 0, 0.3)", "": ""}


def recompute():
    st.session_state.previous = st.session_state.get("recs")
    st.session_state.recs = {model: recommend(st.session_state.basket, model, k=10) for model in models}


if "basket" not in st.session_state:
    st.session_state.basket = []
    recompute()


def compare(new, old):
    table = new[["product_id", "product_name", "aisle", "score"]].assign(status="")
    if old is None:
        return table
    old_position = dict(zip(old["product_id"], old["position"]))
    table["status"] = ["new" if p not in old_position else "moved" if old_position[p] != pos else ""
                       for p, pos in zip(new["product_id"], new["position"])]
    removed = old[~old["product_id"].isin(new["product_id"])]
    removed = removed[["product_id", "product_name", "aisle", "score"]].assign(status="removed")
    return pd.concat([table, removed], ignore_index=True)


def toggle_product():
    product = st.session_state.search
    if product is None:
        return
    st.session_state.search = None
    basket = st.session_state.basket
    if product in basket:
        basket.remove(product)
    else:
        basket.append(product)


tab_recs, tab_quality = st.tabs(["Рекомендації", "Якість моделей"])

with tab_recs:
    chosen_models = st.multiselect("Моделі", models, default=models)
    buttons, captions = st.columns([1, 4])
    with buttons:
        if st.button("Очистити кошик"):
            st.session_state.basket = []
        st.button("Перерахувати", type="primary", on_click=recompute)
    with captions:
        st.caption("Рекомендації оновлюються кнопкою «Перерахувати». Зелений - новий товар у топ-10, "
                   "жовтий - змінив позицію, червоний (знизу) - випав з топ-10.")
        st.caption("Score порівнюваний лише всередині однієї колонки: у кожної моделі своя шкала. "
                   "Score 0 означає доповнення популярними товарами.")
    st.selectbox("Пошук товару (обраний товар додається в кошик або прибирається з нього)",
                 list(names), format_func=names.get, index=None, key="search", on_change=toggle_product)

    basket = st.session_state.basket
    columns = st.columns(1 + len(chosen_models))
    with columns[0]:
        st.subheader("Кошик")
        for product in basket:
            st.write(names[product])

    for column, model in zip(columns[1:], chosen_models):
        with column:
            st.subheader(model)
            previous = st.session_state.previous
            table = compare(st.session_state.recs[model], previous[model] if previous else None)
            styled = table.style.apply(lambda row: [f"background-color: {COLORS[row['status']]}"] * len(row), axis=1)
            st.dataframe(styled, column_order=["product_name", "aisle", "score"], hide_index=True,
                         height="content")

with tab_quality:
    quality = json.loads((ARTIFACTS_DIR / "quality.json").read_text())
    st.dataframe(pd.DataFrame(quality["metrics"]).T)
    st.write(f"Тестових кошиків: {quality['n_test_baskets']}, "
             f"каталог: {quality['catalog_size']} товарів, "
             f"покриття прихованих товарів каталогом: {quality['coverage']:.1%}")
    st.markdown("""
**Протокол:** кошики test (інші користувачі, ніж у train). Видимі = перша половина товарів за порядком
додавання в кошик, приховані = друга половина. Моделі отримують видиму частину, рахуємо Recall@10 і NDCG@10
по прихованих товарах. Беремо лише кошики з 2+ товарами.

**Обмеження:** метрика винагороджує популярні товари і не бачить, чи рекомендація збільшила б чек.
Товари поза каталогом (менше 5 появ у train) моделі рекомендувати не можуть.
""")
