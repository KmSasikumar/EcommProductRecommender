# load_products_to_db.py
import sqlite3
import json
import os

# --- Configuration ---
DATABASE_PATH = "ecommerce.db"
PRODUCTS_JSON_PATH = "products.json" # Path to your existing products.json file

def load_products_to_database():
    """
    Reads products from products.json and inserts/updates them in the SQLite database.
    """
    if not os.path.exists(PRODUCTS_JSON_PATH):
        print(f"Error: Products file {PRODUCTS_JSON_PATH} not found.")
        return

    try:
        with open(PRODUCTS_JSON_PATH, 'r') as f:
            products_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from {PRODUCTS_JSON_PATH}: {e}")
        return
    except Exception as e:
        print(f"Unexpected error reading {PRODUCTS_JSON_PATH}: {e}")
        return

    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        inserted_or_updated_count = 0

        for product in products_data:
            product_id = product.get('id')
            name = product.get('name')
            price = product.get('price')
            category = product.get('category')
            # Convert lists to JSON strings for storage in TEXT columns
            image_urls_json = json.dumps(product.get('imageUrls', []))
            tags_json = json.dumps(product.get('tags', []))

            if not all([product_id, name, price is not None]): # Basic validation (price can be 0)
                print(f"Skipping invalid product data: {product}")
                continue

            # Use INSERT OR REPLACE to handle existing products
            # This will INSERT if the ID is new, or UPDATE if the ID already exists
            try:
                c.execute('''
                    INSERT OR REPLACE INTO products (id, name, price, category, image_urls, tags)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (product_id, name, price, category, image_urls_json, tags_json))
                inserted_or_updated_count += 1

            except sqlite3.Error as e:
                print(f"Database error inserting/updating product {product_id}: {e}")

        conn.commit()
        print(f"Finished loading products. Processed {inserted_or_updated_count} products (including updates).")

    except Exception as e:
        print(f"Unexpected error during database operation: {e}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    load_products_to_database()