# Basket cross-sell

Рекомендації товарів до кошика на даних Instacart: popularity, co-occurrence (conditional, cosine, lift) і EASE.

## Структура
- `scripts/` - підготовка даних і навчання/оцінювання моделей
- `notebooks/` - EDA і 5 прикладів кошиків
- `src/` - код моделей, метрик і завантаження даних
- `app/dashboard.py` - дашборд
- `NOTES.md` - рішення, експерименти і результати

## Запуск
1. Створити середовище, активувати його і встановити залежності (Python 3.12):
   ```
   python -m venv .venv
   .venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
   pip install -r requirements.txt
   ```
   Усі наступні команди виконуються в активованому середовищі.
2. Завантажити дані з https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis і розпакувати csv у `data/raw/`
3. Виконати ноутбуки по черзі (в IDE або з консолі, з папки ноутбука, бо шляхи в ноутбуках відносні):
   ```
   cd scripts
   jupyter nbconvert --to notebook --execute --inplace prepare_data.ipynb
   jupyter nbconvert --to notebook --execute --inplace evaluate.ipynb
   cd ..
   ```
   - `prepare_data.ipynb` - семпл і поділ train/val/test -> `data/processed/`
   - `evaluate.ipynb` - підбір параметрів, метрики на test, збереження моделей -> `artifacts/` (~40 хв)
   - `notebooks/eda.ipynb`, `notebooks/examples.ipynb` - за бажанням
4. Дашборд (з кореня проєкту):
   ```
   streamlit run app/dashboard.py
   ```
