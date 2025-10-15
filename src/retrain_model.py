# retrain_model.py
import pandas as pd
import sqlite3
import os
import numpy as np  # Import numpy for negative sampling logic if needed

# --- NEW: Absolute imports for train.py functions ---
from train import load_and_preprocess_data, train_model, save_mappings, load_mappings

# --- Configuration ---
ORIGINAL_DATA_PATH = "dummy_interactions.csv"
DATABASE_PATH = "user_interactions.db"
MODELS_BASE_DIR = "models_store"
API_KEY_TO_UPDATE = "testkey123"  # The API key whose model we want to update

# --- NEW: Define Interaction Weights ---
INTERACTION_WEIGHTS = {
    "tap": 1.0,      # Basic interest
    "cart": 2.5,     # Stronger intent to purchase
    # "view": 0.5,    # Passive view (example)
    # "like": 1.5,    # Positive sentiment (example)
    # "purchase": 5.0 # Definitive action (if tracked)
}

def retrain_model_with_new_data():
    """
    Reads new interactions from the database, combines with original data (if available),
    assigns weighted scores, and retrains the NCF model for the specified API key.
    The model and mappings are always saved to the SAME paths that the API expects.
    """
    print("Starting model retraining process...")

    # --- 1. Read New Interactions from Database ---
    print(f"Reading new interactions from {DATABASE_PATH}...")
    if not os.path.exists(DATABASE_PATH):
        print(f"Database file {DATABASE_PATH} not found. Skipping retraining.")
        return

    conn = sqlite3.connect(DATABASE_PATH)
    query = "SELECT user_id, item_id, type, timestamp FROM user_interactions"
    try:
        new_interactions_df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error reading from database: {e}")
        conn.close()
        return
    finally:
        conn.close()

    print(f"Found {len(new_interactions_df)} new interactions.")

    if len(new_interactions_df) == 0:
        print("No new interactions found. Skipping retraining.")
        return

    # --- 2. Prepare New Interaction Data (WITH WEIGHTED SCORES) ---
    new_interactions_df['interaction_score'] = new_interactions_df['type'].apply(
        lambda t: INTERACTION_WEIGHTS.get(t, 1.0)
    )
    print(f"Assigned weighted scores to new interactions based on type: {INTERACTION_WEIGHTS}")

    new_interactions_df_renamed = new_interactions_df[['user_id', 'item_id', 'interaction_score']].copy()

    # --- 3. LOAD AND COMBINE WITH ORIGINAL DATA ---
    original_interactions_df = pd.DataFrame()
    if os.path.exists(ORIGINAL_DATA_PATH):
        print(f"Loading original data from {ORIGINAL_DATA_PATH}...")
        try:
            original_interactions_df = pd.read_csv(ORIGINAL_DATA_PATH)
            print(f"Loaded {len(original_interactions_df)} original interactions.")
        except Exception as e:
            print(f"Error loading original data from {ORIGINAL_DATA_PATH}: {e}")
            print("Proceeding with ONLY new interactions for training.")
    else:
        print(f"WARNING: Original data file {ORIGINAL_DATA_PATH} not found!")
        print("         Proceeding with ONLY new interactions for training.")
        print("         Model might forget initial patterns if original data isn't included.")

    # --- 4. COMBINE DATASETS ---
    temp_csv_path = "temp_combined_interactions_for_training.csv"
    try:
        if original_interactions_df.empty:
            combined_df = new_interactions_df_renamed
            print("No original data loaded. Using only new interactions for combined dataset.")
        else:
            original_interactions_df_reset = original_interactions_df.reset_index(drop=True)
            new_interactions_df_reset = new_interactions_df_renamed.reset_index(drop=True)
            combined_df = pd.concat([original_interactions_df_reset, new_interactions_df_reset], ignore_index=True, sort=False)
            print(f"Combined dataset created with {len(combined_df)} total interactions ({len(original_interactions_df_reset)} original + {len(new_interactions_df_reset)} new).")

        # --- 5. Preprocess Combined Data ---
        print("Preprocessing combined data...")
        combined_df.to_csv(temp_csv_path, index=False)
        df_processed, user_map, item_map, num_users, num_items = load_and_preprocess_data(temp_csv_path)

        # --- Crucial: Save to the SAME paths used by the API ---
        model_save_path = os.path.join(MODELS_BASE_DIR, API_KEY_TO_UPDATE, "ncf_model.h5")
        mappings_save_path = os.path.join(MODELS_BASE_DIR, API_KEY_TO_UPDATE, "ncf_mappings.json")

        print(f"Training new model using combined data with weighted interaction scores...")
        trained_model, training_history = train_model(
            df_processed, num_users, num_items,
            model_save_path=model_save_path
        )

        # Overwrite the old model file and mappings so API picks up new ones
        trained_model.save(model_save_path)
        save_mappings(user_map, item_map, mappings_save_path)

        print(f"Model training completed. Saved to {model_save_path}")
        print(f"Retrained model saved to {model_save_path}")
        print(f"Retrained mappings saved to {mappings_save_path}")

        print(f"Updated model for API key '{API_KEY_TO_UPDATE}' with {num_users} users and {num_items} items.")

    except Exception as e:
        print(f"Error during data combination, preprocessing, or training: {e}")
        return
    finally:
        if 'temp_csv_path' in locals() and os.path.exists(temp_csv_path):
            try:
                os.remove(temp_csv_path)
            except OSError as e:
                print(f"Warning: Could not remove temporary file {temp_csv_path}: {e}")

    print("Model retraining process finished successfully!")

# --- Entry Point ---
if __name__ == "__main__":
    retrain_model_with_new_data()
