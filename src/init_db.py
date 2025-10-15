# init_db.py
import sqlite3
import os

# --- Configuration ---
# Paths for the SQLite database files
INTERACTIONS_DB_PATH = "user_interactions.db"
PRODUCTS_DB_PATH = "ecommerce.db" # Or use a single DB file if preferred

def init_interactions_db():
    """
    Creates the user_interactions table in the SQLite database if it doesn't exist.
    """
    conn = sqlite3.connect(INTERACTIONS_DB_PATH)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT NOT NULL,
            item_id TEXT NOT NULL,
            type TEXT NOT NULL, -- e.g., 'tap', 'cart'
            timestamp REAL NOT NULL -- Unix timestamp
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database initialized. Table 'user_interactions' ensured to exist in {INTERACTIONS_DB_PATH}")

def init_products_db():
    """
    Creates the products table in the SQLite database if it doesn't exist.
    """
    conn = sqlite3.connect(PRODUCTS_DB_PATH) # Connect to the products database
    c = conn.cursor()

    # Create the products table
    # Adjust data types and constraints as needed.
    # TEXT for flexibility with lists (tags/image_urls) or use JSON serialization.
    c.execute('''
        CREATE TABLE IF NOT EXISTS products (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            price REAL NOT NULL,
            category TEXT,
            image_urls TEXT, -- Store as JSON stringified list
            tags TEXT         -- Store as JSON stringified list
        )
    ''')

    conn.commit()
    print(f"Database initialized. Table 'products' ensured to exist in {PRODUCTS_DB_PATH}")
    conn.close()

def init_all_databases():
    """
    Initializes all required databases and tables.
    """
    print("Initializing all databases...")
    init_interactions_db()
    init_products_db() # <-- MAKE SURE THIS LINE IS PRESENT
    print("All databases initialized.")

if __name__ == "__main__":
    init_all_databases() # <-- MAKE SURE THIS CALLS init_all_databases