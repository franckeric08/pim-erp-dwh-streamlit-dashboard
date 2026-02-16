import sqlite3
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# ============================================================
# Paths (robuste local + Streamlit Cloud)
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]
DB_PATH = str(BASE_DIR / "db" / "warehouse.db")
RAW_PRICES_CSV = str(BASE_DIR / "data" / "raw" / "erp_prices.csv")

# ============================================================
# Page config
# ============================================================

st.set_page_config(page_title="PIM → ERP → DWH → BI", layout="wide")
st.title("📦 Mini-projet Data : PIM → ERP → DWH → BI")
st.caption("DWH SQLite (schéma étoile) + dashboard Streamlit (Plotly)")

# ============================================================
# Helpers
# ============================================================

@st.cache_data
def load_table(query: str, params=None) -> pd.DataFrame:
    params = params or ()
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query(query, conn, params=params)
    finally:
        conn.close()
    return df


@st.cache_data
def load_dim_options():
    products = load_table("SELECT DISTINCT category, brand FROM dim_product")
    stores = load_table("SELECT DISTINCT store_name FROM dim_store ORDER BY store_name")
    segments = load_table("SELECT DISTINCT segment FROM dim_customer ORDER BY segment")

    return (
        sorted(products["category"].dropna().unique().tolist()),
        sorted(products["brand"].dropna().unique().tolist()),
        stores["store_name"].dropna().unique().tolist(),
        segments["segment"].dropna().unique().tolist(),
    )


@st.cache_data
def date_bounds():
    df = load_table("SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM dim_date")
    min_d = pd.to_datetime(df.loc[0, "min_date"]).date()
    max_d = pd.to_datetime(df.loc[0, "max_date"]).date()
    return min_d, max_d


def fmt_eur(x):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return str(x)


# ============================================================
# Sidebar filters
# ============================================================

cats, brands, stores, segments = load_dim_options()
min_d, max_d = date_bounds()

st.sidebar.header("🔎 Filtres globaux")

date_range = st.sidebar.date_input(
    "Période",
    value=(max(min_d, max_d - timedelta(days=90)), max_d),
    min_value=min_d,
    max_value=max_d,
)

if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = date_range
else:
    d1, d2 = min_d, max_d

# ============================================================
# Load Sales
# ============================================================

sales = load_table("""
    SELECT
        fs.order_id,
        dd.date,
        dc.segment,
        ds.store_name,
        dp.product_name,
        dp.category,
        dp.brand,
        fs.qty,
        fs.unit_price,
        fs.revenue,
        fs.purchase_price,
        fs.margin,
        fs.margin_rate
    FROM fact_sales fs
    JOIN dim_date dd ON dd.date_key = fs.date_key
    JOIN dim_customer dc ON dc.customer_id = fs.customer_id
    JOIN dim_store ds ON ds.store_id = fs.store_id
    JOIN dim_product dp ON dp.product_id = fs.product_id
    WHERE dd.date BETWEEN ? AND ?
""", (str(d1), str(d2)))


# ============================================================
# Executive KPIs
# ============================================================

if sales.empty:
    st.warning("Aucune donnée disponible.")
    st.stop()

total_rev = float(sales["revenue"].sum())
total_margin = float(sales["margin"].sum())
margin_rate = total_margin / total_rev if total_rev else 0
orders = sales["order_id"].nunique()
basket = total_rev / orders if orders else 0

col1, col2, col3, col4 = st.columns(4)
col1.metric("Chiffre d'affaires", fmt_eur(total_rev))
col2.metric("Marge", fmt_eur(total_margin))
col3.metric("Taux de marge", f"{margin_rate*100:.1f}%")
col4.metric("Panier moyen", fmt_eur(basket))

st.divider()

# ============================================================
# Graphiques
# ============================================================

sales_daily = sales.groupby("date").agg(
    revenue=("revenue", "sum"),
    margin=("margin", "sum")
).reset_index()

sales_daily["date"] = pd.to_datetime(sales_daily["date"])

left, right = st.columns(2)
left.plotly_chart(
    px.line(sales_daily, x="date", y="revenue", title="CA journalier"),
    use_container_width=True
)
right.plotly_chart(
    px.line(sales_daily, x="date", y="margin", title="Marge journalière"),
    use_container_width=True
)

st.divider()

top_prod = sales.groupby(["product_name", "category"]).agg(
    revenue=("revenue", "sum"),
    margin=("margin", "sum")
).reset_index().sort_values("revenue", ascending=False)

st.plotly_chart(
    px.bar(top_prod.head(15), x="product_name", y="revenue", title="Top Produits (CA)"),
    use_container_width=True
)

st.dataframe(top_prod.head(20), use_container_width=True)
