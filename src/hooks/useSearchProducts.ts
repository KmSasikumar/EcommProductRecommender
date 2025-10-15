// hooks/useSearchProducts.ts
import { useQuery } from '@tanstack/react-query';
// import { searchProducts } from '../services/mockProducts'; // Remove this import
import { Product } from '../models/Product'; // Keep this import

// Define the structure of the API response
interface SearchApiResponse {
  products: Product[];
}

// This hook will be used for the live search functionality, calling the backend API
// It fetches data whenever the searchQuery changes.
export const useSearchProducts = (searchQuery: string) => {
  return useQuery<Product[], Error>({
    queryKey: ['searchProducts', searchQuery], // Include query in key for caching
    queryFn: async () => {
      // Make the API call to the new Python endpoint
      const response = await fetch(`http://10.22.134.152:8000/search`, {
        method: 'POST', // Or 'GET' with query as a parameter if you change the Python endpoint
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ query: searchQuery }), // Send query in request body
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Search API error (${response.status}):`, errorText);
        throw new Error(`Search API error: ${response.status} ${response.statusText} - ${errorText}`);
      }

      const data: SearchApiResponse = await response.json();
      return data.products; // Return the products array from the response
    },
    // Enable the query only if searchQuery is not empty.
    // This prevents unnecessary calls when the app first loads or search is cleared.
    // If you want the initial list when searchQuery is empty,
    // you might need a separate hook (useProducts) or modify this logic.
    // For now, let's keep it enabled but rely on the backend to handle empty query.
    enabled: true, // Always enabled, backend handles empty query
    staleTime: 1000, // Consider data stale after 1 second
  });
};