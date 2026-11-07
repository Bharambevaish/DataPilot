"""
Create the two sample datasets DataPilot ships with.

Run it from the project root:

    python scripts/make_samples.py

It writes:
  data/samples/sales.csv          ~5,000 rows, one flat table
  data/samples/shop_orders.xlsx   2 sheets (customers, orders) so questions need JOINs

The data is synthetic but shaped like real retail data: weekday/weekend
patterns, a December peak, different price points per category, and one
deliberate story -- the South region falls away in the last quarter of 2024 --
so the demo question "which region's sales dropped the most last quarter?"
has a real, checkable answer.

The random seed is fixed, so re-running this script reproduces the exact same
data. That matters because the evaluation suite stores expected answers.

(Small caveat: sales.csv comes out byte-for-byte identical every time, but
shop_orders.xlsx does not, because openpyxl stamps the creation time into the
.xlsx zip. The rows inside are identical -- only the timestamp differs -- so
git may show the file as changed after a re-run even though the data has not.)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# Same seed every run -> same data every run.
RNG = np.random.default_rng(42)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "samples"

N_SALES_ROWS = 6_900
N_CUSTOMERS = 300
N_ORDERS = 2_000

# Each region has its own cities and its own baseline size, so totals differ.
REGIONS = {
    "North": {"cities": ["Delhi", "Chandigarh", "Jaipur"], "weight": 1.25},
    "South": {"cities": ["Bengaluru", "Chennai", "Hyderabad"], "weight": 1.10},
    "East": {"cities": ["Kolkata", "Patna", "Bhubaneswar"], "weight": 0.80},
    "West": {"cities": ["Mumbai", "Pune", "Ahmedabad"], "weight": 1.00},
}

# product -> (category, typical unit price in rupees)
PRODUCTS = {
    "Laptop Pro 14": ("Electronics", 82_000),
    "Wireless Mouse": ("Electronics", 1_200),
    "Noise Cancelling Headphones": ("Electronics", 9_500),
    "Smart Watch": ("Electronics", 14_000),
    "Office Chair": ("Furniture", 11_500),
    "Standing Desk": ("Furniture", 24_000),
    "Bookshelf": ("Furniture", 6_800),
    "Cotton T-Shirt": ("Apparel", 699),
    "Running Shoes": ("Apparel", 3_499),
    "Winter Jacket": ("Apparel", 4_200),
    "Coffee Beans 1kg": ("Grocery", 950),
    "Olive Oil 1L": ("Grocery", 780),
    "Protein Powder": ("Grocery", 2_400),
}


def _season_multiplier(dates: pd.Series) -> np.ndarray:
    """
    Make the calendar feel real.

    * Weekends are busier than weekdays.
    * Sales build slowly through the year and spike in November/December
      (festival and holiday shopping).
    """
    month = dates.dt.month.to_numpy()
    weekday = dates.dt.day_of_week.to_numpy()  # Monday=0 ... Sunday=6

    monthly = np.array([0.85, 0.88, 0.95, 0.98, 1.00, 0.95,
                        0.97, 1.02, 1.05, 1.15, 1.30, 1.35])
    factor = monthly[month - 1]
    factor = factor * np.where(weekday >= 5, 1.20, 1.00)
    return factor


def make_sales() -> pd.DataFrame:
    """Build the flat sales table: one row per order line in 2024."""
    # Random dates across the whole of 2024.
    start = pd.Timestamp("2024-01-01")
    day_offsets = RNG.integers(0, 366, size=N_SALES_ROWS)
    dates = pd.Series(start + pd.to_timedelta(day_offsets, unit="D"))

    # Pick a region, weighted so the regions are genuinely different sizes.
    region_names = list(REGIONS)
    region_weights = np.array([REGIONS[r]["weight"] for r in region_names])
    region_probs = region_weights / region_weights.sum()
    regions = RNG.choice(region_names, size=N_SALES_ROWS, p=region_probs)

    # A city belonging to that region (so region and city stay consistent).
    cities = [str(RNG.choice(REGIONS[r]["cities"])) for r in regions]

    # Products, and the category/price that go with each one.
    product_names = list(PRODUCTS)
    products = RNG.choice(product_names, size=N_SALES_ROWS)
    categories = [PRODUCTS[p][0] for p in products]
    base_prices = np.array([PRODUCTS[p][1] for p in products], dtype=float)

    # Unit price wobbles +/-8% around the typical price.
    unit_price = base_prices * RNG.normal(1.0, 0.08, size=N_SALES_ROWS)
    unit_price = np.round(np.clip(unit_price, base_prices * 0.75, base_prices * 1.25), 2)

    # Quantity: usually 1-3, occasionally a bulk order.
    quantity = RNG.choice([1, 2, 3, 4, 5, 10], size=N_SALES_ROWS,
                          p=[0.42, 0.26, 0.14, 0.09, 0.06, 0.03])

    # Discounts: most sales have none, some have 5-30% off.
    discount = RNG.choice([0.00, 0.05, 0.10, 0.15, 0.20, 0.30], size=N_SALES_ROWS,
                          p=[0.50, 0.18, 0.14, 0.09, 0.06, 0.03])

    frame = pd.DataFrame({
        "order_id": [f"S{100000 + i}" for i in range(N_SALES_ROWS)],
        "date": dates,
        "region": regions,
        "city": cities,
        "product": products,
        "category": categories,
        "quantity": quantity,
        "unit_price": unit_price,
        "discount": discount,
    })

    # ---- Apply seasonality, then the "South declines" story ----------------
    # Seasonality and the story change *how many rows survive*, which is how
    # real demand changes look: fewer orders, not smaller numbers.
    keep_probability = _season_multiplier(frame["date"]) / 1.45

    # The story: South trades normally for three quarters, then loses roughly
    # half its volume in Q4 after (let's say) two big stores closed.
    q4 = frame["date"].dt.quarter == 4
    south = frame["region"] == "South"
    keep_probability = np.where(q4 & south, keep_probability * 0.45, keep_probability)

    # East grows instead, so the answer is not simply "everyone fell".
    east = frame["region"] == "East"
    keep_probability = np.where(q4 & east, keep_probability * 1.25, keep_probability)

    keep = RNG.random(N_SALES_ROWS) < np.clip(keep_probability, 0, 1)
    frame = frame.loc[keep].reset_index(drop=True)

    # Sort by date so the file reads naturally, then renumber the order ids.
    frame = frame.sort_values("date").reset_index(drop=True)
    frame["order_id"] = [f"S{100000 + i}" for i in range(len(frame))]

    # Store the date as plain YYYY-MM-DD text in the CSV.
    frame["date"] = frame["date"].dt.strftime("%Y-%m-%d")
    return frame


def make_customers_and_orders() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build two related tables, so real questions need a JOIN."""
    first_names = ["Aarav", "Diya", "Ishaan", "Meera", "Rohan", "Ananya", "Kabir",
                   "Priya", "Arjun", "Sara", "Vivaan", "Nisha", "Dev", "Tara",
                   "Aditya", "Riya", "Karan", "Zara", "Neel", "Aisha"]
    last_names = ["Sharma", "Patel", "Reddy", "Iyer", "Nair", "Singh", "Gupta",
                  "Mehta", "Bose", "Khan", "Joshi", "Desai", "Rao", "Kulkarni"]

    all_cities = [c for info in REGIONS.values() for c in info["cities"]]
    segments = ["Consumer", "Small Business", "Enterprise"]

    # Customers signed up over two years, so "how long have they been with us"
    # style questions work.
    signup = pd.Timestamp("2023-01-01") + pd.to_timedelta(
        RNG.integers(0, 730, size=N_CUSTOMERS), unit="D"
    )

    names = [
        f"{RNG.choice(first_names)} {RNG.choice(last_names)}"
        for _ in range(N_CUSTOMERS)
    ]

    customers = pd.DataFrame({
        "customer_id": [f"C{1000 + i}" for i in range(N_CUSTOMERS)],
        "customer_name": names,
        "email": [
            f"{n.lower().replace(' ', '.')}{i}@example.com"
            for i, n in enumerate(names)
        ],
        "city": RNG.choice(all_cities, size=N_CUSTOMERS),
        "country": "India",
        "signup_date": signup,
        "segment": RNG.choice(segments, size=N_CUSTOMERS, p=[0.60, 0.28, 0.12]),
    })

    # Orders. Only the first 270 customers ever order, so 30 customers have no
    # orders at all -- that is what makes LEFT JOIN questions interesting.
    ordering_customers = customers["customer_id"].iloc[:270].to_numpy()

    # Enterprise customers order more often: give them extra weight.
    segment_of = dict(zip(customers["customer_id"], customers["segment"]))
    weight = np.array([
        {"Consumer": 1.0, "Small Business": 1.8, "Enterprise": 3.0}[segment_of[c]]
        for c in ordering_customers
    ])
    probs = weight / weight.sum()

    customer_ids = RNG.choice(ordering_customers, size=N_ORDERS, p=probs)

    order_dates = pd.Timestamp("2024-01-01") + pd.to_timedelta(
        RNG.integers(0, 366, size=N_ORDERS), unit="D"
    )

    # Order value depends on the customer's segment.
    base = np.array([
        {"Consumer": 2_500, "Small Business": 9_000, "Enterprise": 40_000}[segment_of[c]]
        for c in customer_ids
    ], dtype=float)
    amount = np.round(base * RNG.lognormal(mean=0.0, sigma=0.55, size=N_ORDERS), 2)

    orders = pd.DataFrame({
        "order_id": [f"O{50000 + i}" for i in range(N_ORDERS)],
        "customer_id": customer_ids,
        "order_date": order_dates,
        "amount": amount,
        "status": RNG.choice(
            ["delivered", "shipped", "cancelled", "returned"],
            size=N_ORDERS, p=[0.74, 0.13, 0.08, 0.05],
        ),
        "channel": RNG.choice(
            ["web", "mobile_app", "store", "phone"],
            size=N_ORDERS, p=[0.44, 0.34, 0.17, 0.05],
        ),
    }).sort_values("order_date").reset_index(drop=True)
    orders["order_id"] = [f"O{50000 + i}" for i in range(len(orders))]

    return customers, orders


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sales = make_sales()
    sales_path = OUT_DIR / "sales.csv"
    sales.to_csv(sales_path, index=False)

    customers, orders = make_customers_and_orders()
    shop_path = OUT_DIR / "shop_orders.xlsx"
    with pd.ExcelWriter(shop_path, engine="openpyxl", date_format="YYYY-MM-DD",
                        datetime_format="YYYY-MM-DD") as writer:
        customers.to_excel(writer, sheet_name="customers", index=False)
        orders.to_excel(writer, sheet_name="orders", index=False)

    # Print a short report so running the script is self-explanatory.
    print(f"Wrote {sales_path.relative_to(PROJECT_ROOT)}")
    print(f"  {len(sales):,} rows x {len(sales.columns)} columns")
    print(f"  dates {sales['date'].min()} to {sales['date'].max()}")
    print(f"  regions: {', '.join(sorted(sales['region'].unique()))}")
    print()
    print(f"Wrote {shop_path.relative_to(PROJECT_ROOT)}")
    print(f"  sheet 'customers': {len(customers):,} rows x {len(customers.columns)} columns")
    print(f"  sheet 'orders':    {len(orders):,} rows x {len(orders.columns)} columns")


if __name__ == "__main__":
    main()
