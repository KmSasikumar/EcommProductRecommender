// hooks/useSearchProducts.ts
import { useQuery } from '@tanstack/react-query';
import { Product } from '../models/Product';

// Define the structure of the API response
interface SearchApiResponse {
  products: Product[];
}

/**
 * This hook fetches search results from the Python API's /search endpoint.
 * It queries the SQLite database (ecommerce.db) for results based on the search term.
 * This provides dynamic, real-time search results fetched directly from the database.
 * @param searchQuery The search term entered by the user.
 * @returns TanStack Query result object containing data, isLoading, error, etc.
 */
export const useSearchProducts = (searchQuery: string) => {
  return useQuery<Product[], Error>({
    queryKey: ['searchProducts', searchQuery], // Include query in key for caching
    queryFn: async () => {
      // Make the API call to the new Python endpoint that queries ecommerce.db
      const response = await fetch(`http://10.22.134.152:8000/search`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Add 'X-API-Key': 'testkey123' header if your /search endpoint requires it
          // 'X-API-Key': 'testkey123',
        },
        body: JSON.stringify({ query: searchQuery }), // Send the search term in request body
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Search API error (${response.status}):`, errorText);
        throw new Error(`Search API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data: SearchApiResponse = await response.json();
      return data.products; // Return the products array from the response
    },
    // Enable the query whenever searchQuery changes, including when it's an empty string.
    // This allows fetching the initial full product list when the search bar is empty (if backend handles it)
    // or filtered results as the user types.
    enabled: true, // Always enabled to react to searchQuery changes
    staleTime: 1000, // Consider data stale after 1 second (adjust as needed)
  });
};