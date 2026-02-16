import os
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

fake = Faker("fr_FR")
random.seed(42)
np.random.seed(42)

OUT_DIR = "data/raw"


def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)


def gen_pim_products(n_products: int = 500) -> pd.DataFrame:
    categories = ["EPI", "Outillage", "Quincaillerie", "Électricité", "Plomberie"]
    brands = ["DeltaPlus", "3M", "Uvex", "Honeywell", "Bosch", "Makita", "Stanley"]
    norms = ["EN 397", "EN 388", "EN 166", "EN 149", "EN ISO 20345", None]

    rows = []
    for i in range(n_products):
        ean = fake.ean(length=13)
        cat = random.choice(categories)
        brand = random.choice(brands)
        norm = random.choice(norms) if cat == "EPI" else None
        rows.append(
            {
                "product_id": i + 1,
                "ean": ean,
                "sku": f"SKU-{i+1:05d}",
                "product_name": f"{cat} - {fake.word().capitalize()} {fake.word().capitalize()}",
                "category": cat,
                "brand": brand,
                "norm": norm,
                "is_active": True,
                "created_at": fake.date_between(start_date="-2y", end_date="today").isoformat(),
            }
        )

    df = pd.DataFrame(rows)

    # Injecte quelques anomalies réalistes
    # 1) doublons EAN
    if n_products >= 10:
        df.loc[5, "ean"] = df.loc[2, "ean"]
        df.loc[9, "ean"] = df.loc[2, "ean"]

    # 2) attribut manquant
    df.loc[15, "category"] = None
    df.loc[25, "product_name"] = None

    # 3) EPI sans norme
    epi_idx = df[df["category"] == "EPI"].index
    if len(epi_idx) > 0:
        df.loc[epi_idx[0], "norm"] = None

    return df


def gen_suppliers(n_suppliers: int = 50) -> pd.DataFrame:
    rows = []
    for i in range(n_suppliers):
        rows.append(
            {
                "supplier_id": i + 1,
                "supplier_name": fake.company(),
                "country": "FR",
                "lead_time_days": int(np.random.randint(2, 21)),
            }
        )
    return pd.DataFrame(rows)


def gen_stores(n_stores: int = 8) -> pd.DataFrame:
    cities = ["Laval", "Le Mans", "Rennes", "Nantes", "Angers", "Tours", "Caen", "Brest"]
    rows = []
    for i in range(n_stores):
        rows.append(
            {
                "store_id": i + 1,
                "store_name": f"Agence {cities[i % len(cities)]}",
                "city": cities[i % len(cities)],
                "region": "Pays de la Loire" if cities[i % len(cities)] in ["Laval", "Le Mans", "Nantes", "Angers"] else "Ouest",
            }
        )
    return pd.DataFrame(rows)


def gen_customers(n_customers: int = 200) -> pd.DataFrame:
    segments = ["BTP", "Industrie", "Collectivités", "Artisans"]
    rows = []
    for i in range(n_customers):
        rows.append(
            {
                "customer_id": i + 1,
                "customer_name": fake.company(),
                "segment": random.choice(segments),
                "city": fake.city(),
            }
        )
    return pd.DataFrame(rows)


def gen_erp_prices(products: pd.DataFrame, suppliers: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in products.iterrows():
        supplier_id = int(np.random.choice(suppliers["supplier_id"]))
        base_price = float(np.round(np.random.uniform(5, 250), 2))
        rows.append(
            {
                "product_id": int(p["product_id"]),
                "supplier_id": supplier_id,
                "purchase_price": base_price,
                "list_price": float(np.round(base_price * np.random.uniform(1.2, 1.8), 2)),
                "currency": "EUR",
                "updated_at": fake.date_between(start_date="-1y", end_date="today").isoformat(),
            }
        )

    df = pd.DataFrame(rows)

    # Anomalie: prix nul
    if len(df) > 0:
        df.loc[3, "purchase_price"] = 0.0

    return df


def gen_erp_stock(products: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, p in products.iterrows():
        for _, s in stores.iterrows():
            stock = int(np.random.poisson(lam=12))
            rows.append(
                {
                    "product_id": int(p["product_id"]),
                    "store_id": int(s["store_id"]),
                    "stock_qty": stock,
                    "safety_stock": int(np.random.randint(2, 10)),
                    "snapshot_date": datetime.today().date().isoformat(),
                }
            )
    df = pd.DataFrame(rows)

    # Anomalie: stock négatif
    if len(df) > 0:
        df.loc[10, "stock_qty"] = -3

    return df


def gen_erp_sales(products: pd.DataFrame, customers: pd.DataFrame, stores: pd.DataFrame, n_orders: int = 1200) -> pd.DataFrame:
    start = datetime.today() - timedelta(days=180)
    rows = []
    for i in range(n_orders):
        order_id = f"SO-{i+1:06d}"
        order_date = (start + timedelta(days=int(np.random.randint(0, 180)))).date().isoformat()
        customer_id = int(np.random.choice(customers["customer_id"]))
        store_id = int(np.random.choice(stores["store_id"]))

        n_lines = int(np.random.randint(1, 6))
        chosen_products = products.sample(n_lines, replace=False)

        for _, p in chosen_products.iterrows():
            qty = int(np.random.randint(1, 8))
            unit_price = float(np.round(np.random.uniform(8, 320), 2))
            rows.append(
                {
                    "order_id": order_id,
                    "order_date": order_date,
                    "customer_id": customer_id,
                    "store_id": store_id,
                    "product_id": int(p["product_id"]),
                    "qty": qty,
                    "unit_price": unit_price,
                }
            )

    df = pd.DataFrame(rows)

    # Anomalie: ligne avec qty = 0
    if len(df) > 0:
        df.loc[5, "qty"] = 0

    return df


def main():
    ensure_dirs()

    pim_products = gen_pim_products(500)
    suppliers = gen_suppliers(50)
    stores = gen_stores(8)
    customers = gen_customers(200)

    erp_prices = gen_erp_prices(pim_products, suppliers)
    erp_stock = gen_erp_stock(pim_products, stores)
    erp_sales = gen_erp_sales(pim_products, customers, stores, 1200)

    pim_products.to_csv(os.path.join(OUT_DIR, "pim_products.csv"), index=False)
    suppliers.to_csv(os.path.join(OUT_DIR, "erp_suppliers.csv"), index=False)
    stores.to_csv(os.path.join(OUT_DIR, "erp_stores.csv"), index=False)
    customers.to_csv(os.path.join(OUT_DIR, "erp_customers.csv"), index=False)
    erp_prices.to_csv(os.path.join(OUT_DIR, "erp_prices.csv"), index=False)
    erp_stock.to_csv(os.path.join(OUT_DIR, "erp_stock.csv"), index=False)
    erp_sales.to_csv(os.path.join(OUT_DIR, "erp_sales.csv"), index=False)

    print("✅ Données générées dans data/raw/")
    print(" - pim_products.csv")
    print(" - erp_suppliers.csv, erp_stores.csv, erp_customers.csv")
    print(" - erp_prices.csv, erp_stock.csv, erp_sales.csv")


if __name__ == "__main__":
    main()

