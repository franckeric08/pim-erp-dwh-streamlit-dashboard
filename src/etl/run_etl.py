import os
import sqlite3
from datetime import datetime

import pandas as pd

RAW_DIR = "data/raw"
DB_PATH = "db/warehouse.db"


def read_csv(name: str) -> pd.DataFrame:
    path = os.path.join(RAW_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Fichier introuvable: {path}")
    return pd.read_csv(path)


def connect_db() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    cur = conn.cursor()

    # Drop (dev only)
    tables = [
        "fact_sales",
        "fact_stock",
        "dim_date",
        "dim_product",
        "dim_customer",
        "dim_store",
        "dim_supplier",
        "dq_issues",
    ]
    for t in tables:
        cur.execute(f"DROP TABLE IF EXISTS {t};")

    # Dimensions
    cur.execute("""
        CREATE TABLE dim_product (
            product_id INTEGER PRIMARY KEY,
            ean TEXT,
            sku TEXT,
            product_name TEXT,
            category TEXT,
            brand TEXT,
            norm TEXT,
            is_active INTEGER,
            created_at TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE dim_supplier (
            supplier_id INTEGER PRIMARY KEY,
            supplier_name TEXT,
            country TEXT,
            lead_time_days INTEGER
        );
    """)

    cur.execute("""
        CREATE TABLE dim_store (
            store_id INTEGER PRIMARY KEY,
            store_name TEXT,
            city TEXT,
            region TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE dim_customer (
            customer_id INTEGER PRIMARY KEY,
            customer_name TEXT,
            segment TEXT,
            city TEXT
        );
    """)

    cur.execute("""
        CREATE TABLE dim_date (
            date_key INTEGER PRIMARY KEY,   -- YYYYMMDD
            date TEXT NOT NULL,
            year INTEGER,
            month INTEGER,
            day INTEGER,
            week INTEGER
        );
    """)

    # Facts
    cur.execute("""
        CREATE TABLE fact_sales (
            sales_id INTEGER PRIMARY KEY AUTOINCREMENT,
            order_id TEXT,
            date_key INTEGER NOT NULL,
            customer_id INTEGER NOT NULL,
            store_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            qty INTEGER,
            unit_price REAL,
            revenue REAL,
            purchase_price REAL,
            margin REAL,
            margin_rate REAL,
            FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
            FOREIGN KEY(customer_id) REFERENCES dim_customer(customer_id),
            FOREIGN KEY(store_id) REFERENCES dim_store(store_id),
            FOREIGN KEY(product_id) REFERENCES dim_product(product_id)
        );
    """)

    cur.execute("""
        CREATE TABLE fact_stock (
            stock_id INTEGER PRIMARY KEY AUTOINCREMENT,
            date_key INTEGER NOT NULL,
            store_id INTEGER NOT NULL,
            product_id INTEGER NOT NULL,
            stock_qty INTEGER,
            safety_stock INTEGER,
            is_below_safety INTEGER,
            FOREIGN KEY(date_key) REFERENCES dim_date(date_key),
            FOREIGN KEY(store_id) REFERENCES dim_store(store_id),
            FOREIGN KEY(product_id) REFERENCES dim_product(product_id)
        );
    """)

    # Data quality issues
    cur.execute("""
        CREATE TABLE dq_issues (
            issue_id INTEGER PRIMARY KEY AUTOINCREMENT,
            issue_ts TEXT,
            source_table TEXT,
            rule_name TEXT,
            severity TEXT,
            record_ref TEXT,
            details TEXT
        );
    """)

    conn.commit()


def ensure_dim_date(conn: sqlite3.Connection, dates: pd.Series) -> None:
    # dates in ISO string 'YYYY-MM-DD'
    df = pd.DataFrame({"date": pd.to_datetime(dates).dt.date.astype(str)}).drop_duplicates()
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["day"] = pd.to_datetime(df["date"]).dt.day
    df["week"] = pd.to_datetime(df["date"]).dt.isocalendar().week.astype(int)
    df["date_key"] = pd.to_datetime(df["date"]).dt.strftime("%Y%m%d").astype(int)
    df = df[["date_key", "date", "year", "month", "day", "week"]].sort_values("date_key")

    df.to_sql("dim_date", conn, if_exists="append", index=False)


def log_issue(conn: sqlite3.Connection, source_table: str, rule: str, severity: str, record_ref: str, details: str) -> None:
    conn.execute(
        """
        INSERT INTO dq_issues(issue_ts, source_table, rule_name, severity, record_ref, details)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (datetime.now().isoformat(timespec="seconds"), source_table, rule, severity, record_ref, details),
    )
    conn.commit()


def run_quality_checks(conn: sqlite3.Connection, pim_products: pd.DataFrame, prices: pd.DataFrame, stock: pd.DataFrame, sales: pd.DataFrame) -> None:
    # 1) EAN duplicates
    dup_eans = pim_products[pim_products["ean"].astype(str).duplicated(keep=False)]
    for _, r in dup_eans.iterrows():
        log_issue(conn, "pim_products", "DUPLICATE_EAN", "HIGH", f"product_id={int(r['product_id'])}", f"ean={r['ean']}")

    # 2) Missing category / name
    miss_cat = pim_products[pim_products["category"].isna()]
    for _, r in miss_cat.iterrows():
        log_issue(conn, "pim_products", "MISSING_CATEGORY", "MEDIUM", f"product_id={int(r['product_id'])}", "category is NULL")

    miss_name = pim_products[pim_products["product_name"].isna()]
    for _, r in miss_name.iterrows():
        log_issue(conn, "pim_products", "MISSING_PRODUCT_NAME", "MEDIUM", f"product_id={int(r['product_id'])}", "product_name is NULL")

    # 3) Purchase price = 0
    bad_price = prices[prices["purchase_price"] <= 0]
    for _, r in bad_price.iterrows():
        log_issue(conn, "erp_prices", "PURCHASE_PRICE_LE_0", "HIGH", f"product_id={int(r['product_id'])}", f"purchase_price={r['purchase_price']}")

    # 4) Stock negative
    neg_stock = stock[stock["stock_qty"] < 0]
    for _, r in neg_stock.iterrows():
        log_issue(conn, "erp_stock", "NEGATIVE_STOCK", "HIGH", f"product_id={int(r['product_id'])},store_id={int(r['store_id'])}", f"stock_qty={r['stock_qty']}")

    # 5) Sales qty <= 0
    bad_qty = sales[sales["qty"] <= 0]
    for _, r in bad_qty.iterrows():
        log_issue(conn, "erp_sales", "QTY_LE_0", "HIGH", f"order_id={r['order_id']}", f"qty={r['qty']}")


def load_dimensions(conn: sqlite3.Connection) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    pim_products = read_csv("pim_products.csv")
    suppliers = read_csv("erp_suppliers.csv")
    stores = read_csv("erp_stores.csv")
    customers = read_csv("erp_customers.csv")

    # Basic clean: keep columns, ensure types
    pim_products["is_active"] = pim_products["is_active"].astype(int)

    pim_products.to_sql("dim_product", conn, if_exists="append", index=False)
    suppliers.to_sql("dim_supplier", conn, if_exists="append", index=False)
    stores.to_sql("dim_store", conn, if_exists="append", index=False)
    customers.to_sql("dim_customer", conn, if_exists="append", index=False)

    return pim_products, suppliers, stores, customers


def load_facts(conn: sqlite3.Connection, pim_products: pd.DataFrame) -> None:
    prices = read_csv("erp_prices.csv")
    stock = read_csv("erp_stock.csv")
    sales = read_csv("erp_sales.csv")

    # Build dim_date from both sales order_date and stock snapshot_date
    all_dates = pd.concat([sales["order_date"], stock["snapshot_date"]], ignore_index=True)
    ensure_dim_date(conn, all_dates)

    # Join prices into sales to compute margin
    prices_small = prices[["product_id", "purchase_price"]].copy()
    sales2 = sales.merge(prices_small, on="product_id", how="left")

    sales2["revenue"] = sales2["qty"] * sales2["unit_price"]
    sales2["margin"] = (sales2["unit_price"] - sales2["purchase_price"]) * sales2["qty"]
    sales2["margin_rate"] = sales2["margin"] / sales2["revenue"].replace({0: None})

    # date_key
    sales2["date_key"] = pd.to_datetime(sales2["order_date"]).dt.strftime("%Y%m%d").astype(int)

    # Order columns to match table
    sales_fact = sales2[
        ["order_id", "date_key", "customer_id", "store_id", "product_id", "qty", "unit_price", "revenue", "purchase_price", "margin", "margin_rate"]
    ].copy()

    sales_fact.to_sql("fact_sales", conn, if_exists="append", index=False)

    # Stock fact
    stock2 = stock.copy()
    stock2["date_key"] = pd.to_datetime(stock2["snapshot_date"]).dt.strftime("%Y%m%d").astype(int)
    stock2["is_below_safety"] = (stock2["stock_qty"] < stock2["safety_stock"]).astype(int)

    stock_fact = stock2[["date_key", "store_id", "product_id", "stock_qty", "safety_stock", "is_below_safety"]].copy()
    stock_fact.to_sql("fact_stock", conn, if_exists="append", index=False)

    # Quality checks (log into dq_issues)
    run_quality_checks(conn, pim_products, prices, stock, sales)


def main():
    # Safety: ensure raw dir exists
    if not os.path.isdir(RAW_DIR):
        raise FileNotFoundError(f"Dossier introuvable: {RAW_DIR}")

    conn = connect_db()
    try:
        create_schema(conn)
        pim_products, suppliers, stores, customers = load_dimensions(conn)
        load_facts(conn, pim_products)
    finally:
        conn.close()

    print(f"✅ DWH créé et alimenté : {DB_PATH}")
    print("✅ Tables: dim_product, dim_supplier, dim_store, dim_customer, dim_date, fact_sales, fact_stock, dq_issues")


if __name__ == "__main__":
    main()
