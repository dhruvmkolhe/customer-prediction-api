"""
Utility script to migrate existing CSV data to SQLite database.
"""

import pandas as pd
import sqlite3
import os

DATA_DIR = "data"
DB_PATH = os.path.join(DATA_DIR, "customer_data.db")
TRANS_CSV = os.path.join(DATA_DIR, "transactions.csv")
RFM_CSV = os.path.join(DATA_DIR, "rfm_features.csv")

def migrate():
    print("=" * 50)
    print("  🚀 MIGRATING CSV DATA TO SQLITE")
    print("=" * 50)
    
    if not os.path.exists(DATA_DIR):
        print(f"Error: Data directory {DATA_DIR} not found.")
        return

    if not os.path.exists(RFM_CSV):
        print(f"Error: RFM CSV not found. Generate data first.")
        return

    print(f"Connecting to database at {DB_PATH}...")
    conn = sqlite3.connect(DB_PATH)
    
    try:
        # Load and Save Transactions
        if os.path.exists(TRANS_CSV):
            print(f"Loading transactions from {TRANS_CSV}...")
            trans_df = pd.read_csv(TRANS_CSV)
            print(f"Saving {len(trans_df)} transactions to SQL...")
            trans_df.to_sql("transactions", conn, if_exists="replace", index=False)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_trans_customer ON transactions(customer_id)")
        else:
            print("Warning: Transactions CSV not found.")

        # Load and Save RFM
        print(f"Loading RFM features from {RFM_CSV}...")
        rfm_df = pd.read_csv(RFM_CSV)
        print(f"Saving {len(rfm_df)} RFM records to SQL...")
        rfm_df.to_sql("rfm_features", conn, if_exists="replace", index=False)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rfm_customer ON rfm_features(customer_id)")
        
        print("\n✅ MIGRATION SUCCESSFUL!")
    except Exception as e:
        print(f"\n❌ MIGRATION FAILED: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    migrate()
