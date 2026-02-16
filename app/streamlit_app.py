import sqlite3
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = "db/warehouse.db"
RAW_PRICES_CSV = "data/raw/erp_prices.csv"  # pour onglet fournisseurs (mapping product -> supplier)


# -----------------------------
# Helpers
# -----------------------------
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
    # Options pour filtres globaux
    products = load_table("SELECT DISTINCT category, brand FROM dim_product")
    stores = load_table("SELECT DISTINCT store_name FROM dim_store ORDER BY store_name")
    segments = load_table("SELECT DISTINCT segment FROM dim_customer ORDER BY segment")
    return (
        sorted([c for c in products["category"].dropna().unique().tolist()]),
        sorted([b for b in products["brand"].dropna().unique().tolist()]),
        stores["store_name"].dropna().unique().tolist(),
        segments["segment"].dropna().unique().tolist(),
    )


@st.cache_data
def date_bounds():
    df = load_table("SELECT MIN(date) AS min_date, MAX(date) AS max_date FROM dim_date")
    min_d = pd.to_datetime(df.loc[0, "min_date"]).date()
    max_d = pd.to_datetime(df.loc[0, "max_date"]).date()
    return min_d, max_d


def apply_common_filters(df: pd.DataFrame, f: dict, has_customer: bool = True) -> pd.DataFrame:
    # Date already filtered in SQL (date between), but safe
    if "category" in df.columns and f["categories"]:
        df = df[df["category"].isin(f["categories"])]
    if "brand" in df.columns and f["brands"]:
        df = df[df["brand"].isin(f["brands"])]
    if "store_name" in df.columns and f["stores"]:
        df = df[df["store_name"].isin(f["stores"])]
    if has_customer and "segment" in df.columns and f["segments"]:
        df = df[df["segment"].isin(f["segments"])]
    return df


def fmt_eur(x):
    try:
        return f"{x:,.0f} €".replace(",", " ")
    except Exception:
        return str(x)


# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(page_title="PIM → ERP → DWH → BI", layout="wide")
st.title("📦 Mini-projet Data : PIM → ERP → DWH → BI")
st.caption("DWH SQLite (schéma étoile) + dashboard Streamlit (Plotly) avec filtres globaux Achats & Marketing")

# -----------------------------
# Sidebar filters (global)
# -----------------------------
cats, brands, stores, segments = load_dim_options()
min_d, max_d = date_bounds()

st.sidebar.header("🔎 Filtres globaux")
date_range = st.sidebar.date_input("Période", value=(max(min_d, max_d - timedelta(days=90)), max_d), min_value=min_d, max_value=max_d)
if isinstance(date_range, tuple) and len(date_range) == 2:
    d1, d2 = date_range
else:
    d1, d2 = min_d, max_d

filters = {
    "date_from": d1,
    "date_to": d2,
    "categories": st.sidebar.multiselect("Catégorie", cats, default=[]),
    "brands": st.sidebar.multiselect("Marque", brands, default=[]),
    "stores": st.sidebar.multiselect("Agence / Magasin", stores, default=[]),
    "segments": st.sidebar.multiselect("Segment client", segments, default=[]),
}

st.sidebar.divider()
show_data = st.sidebar.checkbox("Afficher les données (tables)", value=False)


# -----------------------------
# Data loaders (Sales & Stock)
# -----------------------------
@st.cache_data
def load_sales(date_from, date_to) -> pd.DataFrame:
    q = """
    SELECT
        fs.order_id,
        dd.date,
        fs.customer_id,
        dc.customer_name,
        dc.segment,
        fs.store_id,
        ds.store_name,
        dp.product_id,
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
    """
    return load_table(q, (str(date_from), str(date_to)))


@st.cache_data
def load_stock(date_from, date_to) -> pd.DataFrame:
    q = """
    SELECT
        dd.date,
        ds.store_name,
        dp.product_id,
        dp.product_name,
        dp.category,
        dp.brand,
        fs.stock_qty,
        fs.safety_stock,
        fs.is_below_safety
    FROM fact_stock fs
    JOIN dim_date dd ON dd.date_key = fs.date_key
    JOIN dim_store ds ON ds.store_id = fs.store_id
    JOIN dim_product dp ON dp.product_id = fs.product_id
    WHERE dd.date BETWEEN ? AND ?
    """
    return load_table(q, (str(date_from), str(date_to)))


def get_prices_mapping() -> pd.DataFrame:
    # mapping product -> supplier pour onglet Achats
    try:
        prices = pd.read_csv(RAW_PRICES_CSV)
    except Exception:
        return pd.DataFrame(columns=["product_id", "supplier_id"])
    return prices[["product_id", "supplier_id"]].dropna()


# Load base
sales = load_sales(filters["date_from"], filters["date_to"])
sales = apply_common_filters(sales, filters, has_customer=True)

stock = load_stock(filters["date_from"], filters["date_to"])
stock = apply_common_filters(stock, filters, has_customer=False)

# -----------------------------
# Tabs
# -----------------------------
tabs = st.tabs([
    "🏁 Executive summary",
    "📈 Ventes & marge",
    "📦 Stocks & ruptures",
    "🤝 Achats & fournisseurs",
    "💶 Pricing & cohérence",
    "🧩 Marketing & assortiment",
    "👥 Clients & segmentation (RFM)",
    "🔮 Prévisions (simple)",
    "✅ Data quality",
])

# -----------------------------
# 1) Executive Summary
# -----------------------------
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)

    total_rev = float(sales["revenue"].sum()) if len(sales) else 0.0
    total_margin = float(sales["margin"].sum()) if len(sales) else 0.0
    margin_rate = (total_margin / total_rev) if total_rev else 0.0
    orders = sales["order_id"].nunique() if len(sales) else 0
    customers_n = sales["customer_id"].nunique() if len(sales) else 0
    basket = (total_rev / orders) if orders else 0.0

    below_safety = int(stock["is_below_safety"].sum()) if len(stock) else 0
    stock_rows = len(stock) if len(stock) else 0
    rupture_rate = (below_safety / stock_rows) if stock_rows else 0.0

    col1.metric("Chiffre d'affaires", fmt_eur(total_rev))
    col2.metric("Marge", fmt_eur(total_margin))
    col3.metric("Taux de marge", f"{margin_rate*100:.1f}%")
    col4.metric("Panier moyen", fmt_eur(basket))

    c1, c2, c3 = st.columns(3)
    c1.metric("Nb commandes", f"{orders:,}".replace(",", " "))
    c2.metric("Clients actifs", f"{customers_n:,}".replace(",", " "))
    c3.metric("Ruptures (stock < sécurité)", f"{below_safety:,}".replace(",", " "))

    st.divider()

    left, right = st.columns(2)

    # Top categories
    if len(sales):
        cat_perf = sales.groupby("category", dropna=False).agg(
            revenue=("revenue", "sum"),
            margin=("margin", "sum")
        ).reset_index().sort_values("revenue", ascending=False)

        fig = px.bar(cat_perf.head(10), x="category", y="revenue", title="Top catégories (CA)")
        left.plotly_chart(fig, use_container_width=True)

        fig2 = px.bar(cat_perf.head(10), x="category", y="margin", title="Top catégories (Marge)")
        right.plotly_chart(fig2, use_container_width=True)
    else:
        left.info("Pas de ventes sur la période/les filtres.")
        right.info("Pas de ventes sur la période/les filtres.")

    # Alerts quick view
    st.subheader("⚠️ Alertes rapides")
    dq = load_table("""
        SELECT rule_name, COUNT(*) AS n
        FROM dq_issues
        GROUP BY rule_name
        ORDER BY n DESC
    """)
    a1, a2 = st.columns(2)
    a1.write("**Data quality (global)**")
    a1.dataframe(dq, use_container_width=True, hide_index=True)
    a2.write("**Ruptures (période filtrée)**")
    if len(stock):
        top_rupt = stock[stock["is_below_safety"] == 1].groupby(["store_name", "category"]).size().reset_index(name="n").sort_values("n", ascending=False)
        a2.dataframe(top_rupt.head(15), use_container_width=True, hide_index=True)
    else:
        a2.info("Pas de données stock sur la période/les filtres.")

# -----------------------------
# 2) Ventes & Marge
# -----------------------------
with tabs[1]:
    st.subheader("📈 Ventes & marge")
    if len(sales) == 0:
        st.warning("Aucune vente pour la période et les filtres sélectionnés.")
    else:
        sales_daily = sales.groupby("date").agg(
            revenue=("revenue", "sum"),
            margin=("margin", "sum"),
            orders=("order_id", "nunique")
        ).reset_index()
        sales_daily["date"] = pd.to_datetime(sales_daily["date"])

        left, right = st.columns(2)
        left.plotly_chart(px.line(sales_daily, x="date", y="revenue", title="CA journalier"), use_container_width=True)
        right.plotly_chart(px.line(sales_daily, x="date", y="margin", title="Marge journalière"), use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)

        top_prod = sales.groupby(["product_name", "category", "brand"]).agg(
            revenue=("revenue", "sum"),
            margin=("margin", "sum"),
            qty=("qty", "sum")
        ).reset_index().sort_values("revenue", ascending=False)

        c1.plotly_chart(px.bar(top_prod.head(15), x="product_name", y="revenue", title="Top produits (CA)"), use_container_width=True)
        c2.plotly_chart(px.bar(top_prod.head(15), x="product_name", y="margin", title="Top produits (Marge)"), use_container_width=True)

        if show_data:
            st.dataframe(top_prod.head(50), use_container_width=True, hide_index=True)

# -----------------------------
# 3) Stocks & Ruptures
# -----------------------------
with tabs[2]:
    st.subheader("📦 Stocks & ruptures")
    if len(stock) == 0:
        st.warning("Aucune donnée stock pour la période et les filtres sélectionnés.")
    else:
        # Ruptures par magasin
        rupt_store = stock.groupby("store_name").agg(
            ruptures=("is_below_safety", "sum"),
            lignes=("is_below_safety", "count")
        ).reset_index()
        rupt_store["rupture_rate"] = rupt_store["ruptures"] / rupt_store["lignes"]

        left, right = st.columns(2)
        left.plotly_chart(px.bar(rupt_store.sort_values("ruptures", ascending=False), x="store_name", y="ruptures", title="Ruptures par agence"), use_container_width=True)
        right.plotly_chart(px.bar(rupt_store.sort_values("rupture_rate", ascending=False), x="store_name", y="rupture_rate", title="Taux de rupture par agence"), use_container_width=True)

        st.divider()

        # Produits à risque
        risk = stock.groupby(["product_name", "category", "brand"]).agg(
            ruptures=("is_below_safety", "sum"),
            avg_stock=("stock_qty", "mean"),
            avg_safety=("safety_stock", "mean"),
        ).reset_index().sort_values("ruptures", ascending=False)

        st.plotly_chart(px.bar(risk.head(20), x="product_name", y="ruptures", title="Top produits en rupture (stock < sécurité)"), use_container_width=True)
        if show_data:
            st.dataframe(risk.head(50), use_container_width=True, hide_index=True)

# -----------------------------
# 4) Achats & fournisseurs
# -----------------------------
with tabs[3]:
    st.subheader("🤝 Performance achats & fournisseurs")
    if len(sales) == 0:
        st.warning("Aucune vente sur la période/filtres : impossible d’évaluer la performance fournisseurs.")
    else:
        prices_map = get_prices_mapping()
        if prices_map.empty:
            st.info("Le fichier data/raw/erp_prices.csv est introuvable : onglet fournisseurs limité.")
        else:
            suppliers = load_table("SELECT supplier_id, supplier_name, lead_time_days FROM dim_supplier")

            # join sales -> supplier via product_id
            s = sales.merge(prices_map, on="product_id", how="left").merge(suppliers, on="supplier_id", how="left")

            perf = s.groupby(["supplier_id", "supplier_name", "lead_time_days"]).agg(
                revenue=("revenue", "sum"),
                margin=("margin", "sum"),
                orders=("order_id", "nunique"),
                products=("product_id", "nunique"),
            ).reset_index()

            perf["margin_rate"] = perf["margin"] / perf["revenue"].replace({0: np.nan})
            perf = perf.sort_values("revenue", ascending=False)

            left, right = st.columns(2)
            left.plotly_chart(px.bar(perf.head(15), x="supplier_name", y="revenue", title="Top fournisseurs (CA)"), use_container_width=True)
            right.plotly_chart(px.bar(perf.head(15), x="supplier_name", y="margin_rate", title="Taux de marge par fournisseur"), use_container_width=True)

            st.divider()

            # Fournisseurs à risque : lead time élevé + marge faible
            perf2 = perf.copy()
            lt_thr = perf2["lead_time_days"].quantile(0.75) if perf2["lead_time_days"].notna().any() else 0
            mr_thr = perf2["margin_rate"].quantile(0.25) if perf2["margin_rate"].notna().any() else 0
            perf2["risk_flag"] = ((perf2["lead_time_days"] >= lt_thr) & (perf2["margin_rate"] <= mr_thr)).astype(int)

            risk = perf2[perf2["risk_flag"] == 1].sort_values(["lead_time_days", "margin_rate"], ascending=[False, True])
            st.write("**📌 Fournisseurs à risque (lead time élevé + marge faible)**")
            st.dataframe(risk[["supplier_name", "lead_time_days", "revenue", "margin", "margin_rate", "products", "orders"]].head(20),
                         use_container_width=True, hide_index=True)

# -----------------------------
# 5) Pricing & cohérence
# -----------------------------
with tabs[4]:
    st.subheader("💶 Pricing & cohérence tarifaire")
    if len(sales) == 0:
        st.warning("Aucune vente sur la période/filtres.")
    else:
        s = sales.copy()
        s["is_loss"] = (s["margin"] < 0).astype(int)

        # alertes pricing
        loss = s[s["margin"] < 0].groupby(["product_name", "category"]).agg(
            loss_orders=("order_id", "nunique"),
            loss_amount=("margin", "sum")
        ).reset_index().sort_values("loss_amount")

        high_m = s[s["margin_rate"] > 0.7].groupby(["product_name", "category"]).agg(
            revenue=("revenue", "sum"),
            margin_rate=("margin_rate", "mean")
        ).reset_index().sort_values("revenue", ascending=False)

        left, right = st.columns(2)
        left.plotly_chart(px.histogram(s, x="margin_rate", title="Distribution des taux de marge"), use_container_width=True)
        right.plotly_chart(px.scatter(
            s.sample(min(3000, len(s))),
            x="unit_price",
            y="purchase_price",
            title="Prix de vente vs prix d'achat (échantillon)",
        ), use_container_width=True)

        st.divider()
        c1, c2 = st.columns(2)
        c1.write("**⚠️ Ventes à perte (top)**")
        c1.dataframe(loss.head(20), use_container_width=True, hide_index=True)
        c2.write("**📌 Marges très élevées (à vérifier)**")
        c2.dataframe(high_m.head(20), use_container_width=True, hide_index=True)

# -----------------------------
# 6) Marketing & assortiment
# -----------------------------
with tabs[5]:
    st.subheader("🧩 Marketing & assortiment")
    if len(sales) == 0:
        st.warning("Aucune vente sur la période/filtres.")
    else:
        prod = sales.groupby(["product_id", "product_name", "category", "brand"]).agg(
            revenue=("revenue", "sum"),
            margin=("margin", "sum"),
            qty=("qty", "sum"),
            orders=("order_id", "nunique"),
        ).reset_index()

        # Quadrants based on median
        rev_med = prod["revenue"].median()
        mar_med = prod["margin"].median()
        prod["quadrant"] = np.select(
            [
                (prod["revenue"] >= rev_med) & (prod["margin"] >= mar_med),
                (prod["revenue"] >= rev_med) & (prod["margin"] < mar_med),
                (prod["revenue"] < rev_med) & (prod["margin"] >= mar_med),
            ],
            ["Stars", "Volume drivers", "Profit makers"],
            default="Dead stock",
        )

        fig = px.scatter(
            prod,
            x="revenue",
            y="margin",
            hover_data=["product_name", "category", "brand", "qty", "orders"],
            color="quadrant",
            title="Matrice assortiment (CA vs Marge)",
        )
        st.plotly_chart(fig, use_container_width=True)

        left, right = st.columns(2)
        top_stars = prod[prod["quadrant"] == "Stars"].sort_values("revenue", ascending=False).head(15)
        dead = prod[prod["quadrant"] == "Dead stock"].sort_values("revenue", ascending=True).head(15)

        left.write("**⭐ Stars (à pousser / sécuriser en stock)**")
        left.dataframe(top_stars[["product_name", "category", "revenue", "margin", "qty"]], use_container_width=True, hide_index=True)

        right.write("**🧊 Dead stock (à rationaliser / promo / déréférencement)**")
        right.dataframe(dead[["product_name", "category", "revenue", "margin", "qty"]], use_container_width=True, hide_index=True)

# -----------------------------
# 7) Clients & Segmentation (RFM)
# -----------------------------
with tabs[6]:
    st.subheader("👥 Clients & segmentation (RFM light)")
    if len(sales) == 0:
        st.warning("Aucune vente sur la période/filtres.")
    else:
        s = sales.copy()
        s["date"] = pd.to_datetime(s["date"])
        last_date = s["date"].max()

        rfm = s.groupby(["customer_id", "customer_name", "segment"]).agg(
            recency_days=("date", lambda x: (last_date - x.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("revenue", "sum"),
            margin=("margin", "sum"),
        ).reset_index()

        # Scores by quartiles (1..4)
        rfm["R"] = pd.qcut(rfm["recency_days"], 4, labels=[4, 3, 2, 1]).astype(int)  # plus récent = meilleur
        rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["M"] = pd.qcut(rfm["monetary"].rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)
        rfm["RFM"] = rfm["R"] + rfm["F"] + rfm["M"]

        # simple segmentation
        rfm["segment_rfm"] = np.select(
            [
                rfm["RFM"] >= 10,
                (rfm["RFM"] >= 7) & (rfm["RFM"] < 10),
                (rfm["RFM"] >= 5) & (rfm["RFM"] < 7),
            ],
            ["VIP", "Réguliers", "À développer"],
            default="À réactiver",
        )

        seg = rfm.groupby("segment_rfm").agg(
            customers=("customer_id", "nunique"),
            revenue=("monetary", "sum"),
            margin=("margin", "sum"),
        ).reset_index().sort_values("revenue", ascending=False)

        left, right = st.columns(2)
        left.plotly_chart(px.bar(seg, x="segment_rfm", y="revenue", title="CA par segment RFM"), use_container_width=True)
        right.plotly_chart(px.bar(seg, x="segment_rfm", y="customers", title="Nombre de clients par segment RFM"), use_container_width=True)

        if show_data:
            st.dataframe(rfm.sort_values("RFM", ascending=False).head(50), use_container_width=True, hide_index=True)

# -----------------------------
# 8) Prévisions (simple)
# -----------------------------
with tabs[7]:
    st.subheader("🔮 Prévisions (simple : moyenne mobile + tendance)")
    if len(sales) == 0:
        st.warning("Aucune vente sur la période/filtres.")
    else:
        daily = sales.groupby("date").agg(revenue=("revenue", "sum")).reset_index()
        daily["date"] = pd.to_datetime(daily["date"])
        daily = daily.sort_values("date")
        daily["ma7"] = daily["revenue"].rolling(7).mean()
        daily["ma30"] = daily["revenue"].rolling(30).mean()

        # tendance linéaire simple
        x = np.arange(len(daily))
        if len(daily) >= 10:
            coef = np.polyfit(x, daily["revenue"].values, 1)
            daily["trend"] = coef[0] * x + coef[1]
        else:
            daily["trend"] = np.nan

        fig = px.line(daily, x="date", y=["revenue", "ma7", "ma30", "trend"], title="CA : réel, moyennes mobiles, tendance")
        st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# 9) Data Quality
# -----------------------------
with tabs[8]:
    st.subheader("✅ Data Quality (issues)")
    dq = load_table("SELECT * FROM dq_issues ORDER BY issue_id DESC")
    if dq.empty:
        st.success("Aucune anomalie détectée (ou table vide).")
    else:
        agg = dq.groupby(["source_table", "rule_name", "severity"]).size().reset_index(name="n").sort_values("n", ascending=False)

        left, right = st.columns(2)
        left.plotly_chart(px.bar(agg, x="rule_name", y="n", color="severity", title="Issues par règle"), use_container_width=True)
        right.plotly_chart(px.bar(agg, x="source_table", y="n", color="severity", title="Issues par source"), use_container_width=True)

        st.divider()
        st.write("**Détail des anomalies**")
        st.dataframe(dq.head(200), use_container_width=True, hide_index=True)
