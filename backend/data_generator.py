"""
Mock Data Generator for Customer Purchase Behavior Prediction
Generates realistic e-commerce transaction data with customer demographics.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import uuid
import sqlite3
import os


class DataGenerator:
    """Generates synthetic e-commerce transaction data."""

    def __init__(self, n_customers=5000, n_transactions=20000, seed=42):
        self.n_customers = n_customers
        self.n_transactions = n_transactions
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)

        # Configuration
        self.categories = ["Electronics", "Clothing", "Home & Kitchen", "Sports", "Books", "Beauty"]
        self.category_weights = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
        self.category_price_ranges = {
            "Electronics": (50, 500),
            "Clothing": (15, 150),
            "Home & Kitchen": (20, 300),
            "Sports": (25, 200),
            "Books": (5, 50),
            "Beauty": (10, 100),
        }
        self.payment_methods = ["Credit Card", "Debit Card", "Cash", "UPI", "Net Banking"]
        self.payment_weights = [0.35, 0.25, 0.15, 0.15, 0.10]
        self.locations = [
            "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
            "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose",
            "Austin", "Jacksonville", "Fort Worth", "Columbus", "Charlotte",
            "Indianapolis", "San Francisco", "Seattle", "Denver", "Washington DC"
        ]

    def _generate_customer_profiles(self) -> pd.DataFrame:
        """Generate customer demographic profiles."""
        customers = []
        start_date = datetime(2022, 1, 1)

        for i in range(self.n_customers):
            customer_id = 10001 + i
            age = int(np.clip(np.random.normal(35, 12), 18, 70))
            gender = random.choices(["M", "F", "Other"], weights=[0.48, 0.48, 0.04])[0]
            location = random.choice(self.locations)
            signup_date = start_date + timedelta(days=random.randint(0, 180))

            # Customer behavior profile (affects purchase patterns)
            activity_level = random.choices(
                ["high", "medium", "low"],
                weights=[0.2, 0.5, 0.3]
            )[0]

            customers.append({
                "customer_id": customer_id,
                "age": age,
                "gender": gender,
                "location": location,
                "signup_date": signup_date,
                "activity_level": activity_level,
            })

        return pd.DataFrame(customers)

    def _generate_transactions(self, customers: pd.DataFrame) -> pd.DataFrame:
        """Generate transaction records based on customer profiles."""
        transactions = []
        end_date = datetime(2024, 12, 31)
        start_date = datetime(2023, 1, 1)

        # Activity level affects transaction frequency
        activity_multipliers = {"high": 1.5, "medium": 1.0, "low": 0.5}

        # Generate transactions distributed across time
        for _ in range(self.n_transactions):
            customer = customers.sample(1).iloc[0]
            customer_id = customer["customer_id"]
            activity = activity_multipliers[customer["activity_level"]]

            # Random transaction date with seasonal patterns
            days_range = (end_date - start_date).days
            random_day = random.randint(0, days_range)
            trans_date = start_date + timedelta(days=random_day)

            # Add seasonal bias (more purchases in Nov-Dec)
            month = trans_date.month
            if month in [11, 12]:
                if random.random() > 0.6:
                    trans_date = start_date + timedelta(
                        days=random.randint(days_range - 60, days_range)
                    )

            # Select category with weights
            category = random.choices(self.categories, weights=self.category_weights)[0]
            price_range = self.category_price_ranges[category]

            # Quantity influenced by category
            if category == "Books":
                quantity = np.random.poisson(2) + 1
            elif category == "Electronics":
                quantity = np.random.poisson(0.5) + 1
            else:
                quantity = np.random.poisson(1.5) + 1

            quantity = min(quantity, 20)

            # Unit price with category-specific distribution
            unit_price = round(np.random.lognormal(
                mean=np.log((price_range[0] + price_range[1]) / 3),
                sigma=0.5
            ), 2)
            unit_price = np.clip(unit_price, price_range[0], price_range[1])

            # Payment method
            payment = random.choices(
                self.payment_methods,
                weights=self.payment_weights
            )[0]

            transactions.append({
                "transaction_id": str(uuid.uuid4())[:8],
                "customer_id": customer_id,
                "transaction_date": trans_date,
                "product_category": category,
                "quantity": quantity,
                "unit_price": unit_price,
                "total_amount": round(quantity * unit_price, 2),
                "payment_method": payment,
            })

        return pd.DataFrame(transactions)

    def _calculate_rfm(self, transactions: pd.DataFrame) -> pd.DataFrame:
        """Calculate RFM (Recency, Frequency, Monetary) features."""
        reference_date = transactions["transaction_date"].max() + timedelta(days=1)

        rfm = transactions.groupby("customer_id").agg(
            recency=("transaction_date", lambda x: (reference_date - x.max()).days),
            frequency=("transaction_id", "nunique"),
            monetary=("total_amount", "sum"),
            avg_order_value=("total_amount", "mean"),
            total_quantity=("quantity", "sum"),
            unique_categories=("product_category", "nunique"),
            first_purchase=("transaction_date", "min"),
            last_purchase=("transaction_date", "max"),
        ).reset_index()

        # Calculate additional features
        rfm["customer_lifetime_days"] = (
            rfm["last_purchase"] - rfm["first_purchase"]
        ).dt.days + 1

        rfm["purchase_frequency"] = (
            rfm["frequency"] / (rfm["customer_lifetime_days"] / 30)
        ).clip(0.1, 50)

        rfm["product_diversity"] = (
            rfm["unique_categories"] / len(self.categories)
        ).round(3)

        # Clean up
        rfm["avg_order_value"] = rfm["avg_order_value"].round(2)
        rfm["purchase_frequency"] = rfm["purchase_frequency"].round(3)
        rfm = rfm.drop(columns=["first_purchase", "last_purchase", "total_quantity", "unique_categories"])

        return rfm

    def generate(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Generate complete mock dataset."""
        print(f"Generating {self.n_customers} customer profiles...")
        customers = self._generate_customer_profiles()

        print(f"Generating {self.n_transactions} transactions...")
        transactions = self._generate_transactions(customers)

        print("Calculating RFM features...")
        rfm = self._calculate_rfm(transactions)

        print(f"Generated {len(customers)} customers, {len(transactions)} transactions, {len(rfm)} RFM records")
        return transactions, rfm

    def save(self, output_dir: str = ".", use_db: bool = True):
        """Generate and save data to CSV and/or SQLite."""
        os.makedirs(output_dir, exist_ok=True)

        transactions, rfm = self.generate()

        # Save to CSV for legacy support
        transactions_path = os.path.join(output_dir, "transactions.csv")
        rfm_path = os.path.join(output_dir, "rfm_features.csv")

        transactions.to_csv(transactions_path, index=False)
        rfm.to_csv(rfm_path, index=False)

        print(f"Saved transactions to {transactions_path}")
        print(f"Saved RFM features to {rfm_path}")

        if use_db:
            db_path = os.path.join(output_dir, "customer_data.db")
            self.save_to_db(transactions, rfm, db_path)

        return transactions, rfm

    def save_to_db(self, transactions: pd.DataFrame, rfm: pd.DataFrame, db_path: str):
        """Save dataframes to SQLite database."""
        print(f"Saving data to SQLite database at {db_path}...")
        conn = sqlite3.connect(db_path)
        
        try:
            # Save transactions
            transactions.to_sql("transactions", conn, if_exists="replace", index=False)
            
            # Save RFM features
            rfm.to_sql("rfm_features", conn, if_exists="replace", index=False)
            
            # Create indexes for performance
            cursor = conn.cursor()
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_trans_customer ON transactions(customer_id)")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_rfm_customer ON rfm_features(customer_id)")
            
            print(f"Successfully saved to database and created indexes.")
        finally:
            conn.close()


if __name__ == "__main__":
    generator = DataGenerator(n_customers=10000, n_transactions=50000)
    transactions, rfm = generator.save("data")
    print("\nSample RFM data:")
    print(rfm.head())
    print("\nRFM Statistics:")
    print(rfm.describe())
