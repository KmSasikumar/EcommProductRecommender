// services/productMapping.ts
import { Product } from '../models/Product';

// --- NEW: Function to fetch product details from the Python API ---
export const fetchProductByItemId = async (itemId: string): Promise<Product | null> => {
  if (!itemId) {
    return null;
  }

  try {
    // Make the API call to the new Python endpoint that queries ecommerce.db
    const response = await fetch(`http://10.22.134.152:8000/search`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        // Add 'X-API-Key': 'testkey123' header if your /search endpoint requires it
        // 'X-API-Key': 'testkey123',
      },
      body: JSON.stringify({ query: itemId }), // Search for the specific item_id
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Search API error (${response.status}):`, errorText);
      return null; // Return null on error
    }

    const data = await response.json();
    
    // Assuming the API returns an array of products
    // Find the product with the matching id
    const product = data.products.find((p: Product) => p.id === itemId);
    
    if (product) {
      return product;
    } else {
      console.warn(`Product with ID '${itemId}' not found in search results.`);
      return null; // Return null if product not found
    }
  } catch (error) {
    // Catch network errors like 'Failed to fetch' or timeouts
    console.error(`Error fetching product '${itemId}' from backend:`, error);
    return null; // Return null on error
  }
};