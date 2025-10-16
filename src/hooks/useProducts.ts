// hooks/useProducts.ts
import { useQuery } from '@tanstack/react-query';
import { Product } from '../models/Product';

// Define the structure of the API response
interface SearchApiResponse {
  products: Product[];
}

/**
 * This hook fetches the initial list of all products from the Python API's /search endpoint.
 * It queries the SQLite database (ecommerce.db) with an empty query to get the full list.
 * This replaces the direct use of mockProducts.ts for the initial load.
 * @returns TanStack Query result object containing data, isLoading, error, etc.
 */
export const useProducts = () => {
  return useQuery<Product[], Error>({
    queryKey: ['products'], // Unique key for caching the initial list
    queryFn: async () => {
      // Make the API call to the new Python endpoint that queries ecommerce.db
      // Send an empty query to get ALL products
      const response = await fetch(`http://10.22.134.152:8000/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add 'X-API-Key': 'testkey123' header if your /search endpoint requires it
          // 'X-API-Key': 'testkey123',
        },
        body: JSON.stringify({ query: "" }), // Send empty query to get all products
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Search API error (${response.status}):`, errorText);
        throw new Error(`Search API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data: SearchApiResponse = await response.json();
      return data.products; // Return the products array from the response
    },
    // Enable the query by default to fetch the initial product list
    enabled: true,
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};