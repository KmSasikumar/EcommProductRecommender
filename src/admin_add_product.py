# admin_add_product.py
import sqlite3
import json
import os

# --- Configuration ---
DATABASE_PATH = "ecommerce.db" # Path to your SQLite database file

def get_product_details(initial_data=None):
    """
    Prompts the user to enter product details via the command line.
    If initial_data is provided, it pre-fills the prompts (for editing).
    Returns:
        A dictionary containing the product details.
    """
    print("--- Enter Product Details ---")
    
    # Get Product ID
    if initial_data and 'id' in initial_data:
        product_id = initial_data['id']
        print(f"Product ID (cannot change): {product_id}")
    else:
        product_id = input("Enter unique Product ID (e.g., item5): ").strip()
        if not product_id:
            raise ValueError("Product ID cannot be empty.")

    # Get Name
    default_name = initial_data.get('name', '') if initial_data else ''
    name_prompt = f"Enter Product Name (default: {default_name}): " if default_name else "Enter Product Name: "
    name = input(name_prompt).strip() or default_name
    if not name:
        raise ValueError("Product Name cannot be empty.")

    # Get Price
    default_price = initial_data.get('price', '') if initial_data else ''
    price_prompt = f"Enter Price (e.g., 29.99) (default: {default_price}): " if default_price else "Enter Price (e.g., 29.99): "
    price_input = input(price_prompt).strip() or str(default_price)
    try:
        price = float(price_input) if price_input else 0.0
        if price < 0:
             raise ValueError("Price cannot be negative.")
    except ValueError as e:
        if "could not convert" in str(e):
            raise ValueError("Invalid price entered. Please enter a number.")
        else:
            raise e # Re-raise if it's our custom negative check

    # Get Category
    default_category = initial_data.get('category', '') if initial_data else ''
    category_prompt = f"Enter Category (e.g., Electronics) (default: {default_category}): " if default_category else "Enter Category (e.g., Electronics): "
    category = input(category_prompt).strip() or default_category
    if not category:
        raise ValueError("Category cannot be empty.")

    # Get Image URLs
    print("Enter Image URLs (one per line). Press Enter twice when done:")
    image_urls = []
    if initial_data and 'image_urls' in initial_data:
        print("Current Image URLs:")
        for i, url in enumerate(initial_data['image_urls']):
            print(f"  {i+1}. {url}")
        print("Enter new URLs to replace them, or just press Enter twice to keep current ones:")

    while True:
        url = input("Image URL: ").strip()
        if not url:
            break
        image_urls.append(url)
    
    # If editing and no new URLs entered, keep the old ones
    if initial_data and 'image_urls' in initial_data and not image_urls:
        image_urls = initial_data['image_urls']
        print("Keeping existing image URLs.")

    # Get Tags
    default_tags_str = ", ".join(initial_data.get('tags', [])) if initial_data and 'tags' in initial_data else ""
    tags_prompt = f"Enter Tags (comma-separated, e.g., usb,hub,accessory) (default: {default_tags_str}): " if default_tags_str else "Enter Tags (comma-separated, e.g., usb,hub,accessory): "
    tags_input = input(tags_prompt).strip() or default_tags_str
    tags = [tag.strip() for tag in tags_input.split(',') if tag.strip()]

    return {
        "id": product_id,
        "name": name,
        "price": price,
        "category": category,
        "image_urls": image_urls if image_urls else (initial_data.get('image_urls', []) if initial_data else []),
        "tags": tags if tags else (initial_data.get('tags', []) if initial_data else [])
    }


def insert_or_update_product_in_db(product_data):
    """
    Inserts or updates the product data into the 'products' table.
    Args:
        product_data (dict): Dictionary containing product details.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        image_urls_json = json.dumps(product_data["image_urls"])
        tags_json = json.dumps(product_data["tags"])

        # Use INSERT OR REPLACE to handle updates if ID exists
        c.execute('''
            INSERT OR REPLACE INTO products (id, name, price, category, image_urls, tags)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            product_data["id"],
            product_data["name"],
            product_data["price"],
            product_data["category"],
            image_urls_json,
            tags_json
        ))

        conn.commit()
        action = "added/updated" # Since we use INSERT OR REPLACE
        print(f"\nSuccess! Product '{product_data['name']}' (ID: {product_data['id']}) {action} in '{DATABASE_PATH}'.")

    except sqlite3.Error as e:
        print(f"\nDatabase error occurred: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
    finally:
        if conn:
            conn.close()


def find_product_by_id(product_id):
    """
    Finds a product in the database by its ID.
    Args:
        product_id (str): The ID of the product to find.
    Returns:
        A dictionary representing the product, or None if not found.
    """
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        # Enable row factory to access columns by name
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT * FROM products WHERE id = ?", (product_id,))
        row = c.fetchone()

        if row:
            # Convert sqlite3.Row to dict and parse JSON
            product_dict = dict(row)
            try:
                product_dict['image_urls'] = json.loads(product_dict['image_urls']) if product_dict['image_urls'] else []
                product_dict['tags'] = json.loads(product_dict['tags']) if product_dict['tags'] else []
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON for product {product_dict['id']}. Using empty lists.")
                product_dict['image_urls'] = []
                product_dict['tags'] = []
            return product_dict
        else:
            return None

    except sqlite3.Error as e:
        print(f"Database error while finding product: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred while finding product: {e}")
        return None
    finally:
        if conn:
            conn.close()


def edit_product():
    """
    Allows the user to edit an existing product.
    """
    print("--- Edit Product ---")
    product_id = input("Enter the Product ID to edit: ").strip()
    if not product_id:
        print("Product ID cannot be empty.")
        return

    # 1. Find the product
    product_data = find_product_by_id(product_id)
    if not product_data:
        print(f"Product with ID '{product_id}' not found.")
        return

    print(f"Found product: {product_data['name']}")
    # Pretty print current data? (Optional)

    # 2. Get updated details (pre-filled)
    try:
        updated_data = get_product_details(initial_data=product_data)
        # 3. Update in DB
        insert_or_update_product_in_db(updated_data)
    except ValueError as ve:
        print(f"\nInput Error: {ve}")
    except Exception as e:
        print(f"\nFailed to edit product: {e}")


def delete_product():
    """
    Allows the user to delete an existing product.
    """
    print("--- Delete Product ---")
    product_id = input("Enter the Product ID to delete: ").strip()
    if not product_id:
        print("Product ID cannot be empty.")
        return

    # 1. Confirm deletion
    confirmation = input(f"Are you sure you want to delete product ID '{product_id}'? Type 'yes' to confirm: ").strip().lower()
    if confirmation != 'yes':
        print("Deletion cancelled.")
        return

    # 2. Delete from DB
    conn = None
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()

        c.execute("DELETE FROM products WHERE id = ?", (product_id,))
        rows_affected = c.rowcount
        conn.commit()

        if rows_affected > 0:
            print(f"\nSuccess! Product with ID '{product_id}' deleted from '{DATABASE_PATH}'.")
        else:
            print(f"\nProduct with ID '{product_id}' not found. Nothing was deleted.")

    except sqlite3.Error as e:
        print(f"\nDatabase error occurred during deletion: {e}")
        if conn:
            conn.rollback()
    except Exception as e:
        print(f"\nAn unexpected error occurred during deletion: {e}")
    finally:
        if conn:
            conn.close()


def add_product():
    """
    Wrapper function for the original add product logic.
    """
    print("--- Add New Product ---")
    try:
        product_details = get_product_details() # No initial data for adding
        insert_or_update_product_in_db(product_details)
    except ValueError as ve:
        print(f"\nInput Error: {ve}")
    except Exception as e:
        print(f"\nFailed to add product: {e}")


def main_menu():
    """
    Displays the main menu and handles user choice.
    """
    while True:
        print("\n--- Product Administration Menu ---")
        print("1. Add New Product")
        print("2. Edit Existing Product")
        print("3. Delete Product")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ").strip()

        if choice == '1':
            add_product()
        elif choice == '2':
            edit_product()
        elif choice == '3':
            delete_product()
        elif choice == '4':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please enter a number between 1 and 4.")


def main():
    """
    Main function to run the product administration script.
    """
    if not os.path.exists(DATABASE_PATH):
        print(f"Error: Database file '{DATABASE_PATH}' not found in the current directory.")
        print("Please ensure you are running this script from the correct directory.")
        return

    # Run the main menu loop
    main_menu()


if __name__ == "__main__":
    main()