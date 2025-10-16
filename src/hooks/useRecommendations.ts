// hooks/useRecommendations.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

// Define the structure matching the API's RecommendationItem
interface RecommendationItemFromAPI {
  item_id: string;
  score: number;
}

interface RecommendationResponse {
  recommendations: RecommendationItemFromAPI[];
  user_id: string;
}

// Define the structure for the request payload
interface RecommendationRequestPayload {
  user_id: string;
  count: number;
  search_query: string;
}

export const useRecommendations = (searchQuery: string) => {
  console.log('[useRecommendations] Hook called with searchQuery:', searchQuery);
  
  return useQuery<RecommendationResponse, Error>({
    queryKey: ['recs', searchQuery],
    queryFn: async () => {
      console.log('[useRecommendations] Making API call with query:', searchQuery);
      
      try {
        const response = await axios({
          method: 'post',
          url: 'http://10.22.134.152:8000/recommend',
          headers: {
            'X-API-Key': 'testkey123',
            'Content-Type': 'application/json',
          },
          data: {
            user_id: 'user0',
            count: 10,
            search_query: searchQuery, // Send the actual search query to backend
          } as RecommendationRequestPayload,
        });

        console.log('[useRecommendations] API Response:', response.data);
        console.log('[useRecommendations] Recommendations count:', response.data.recommendations?.length || 0);
        
        // Log each recommendation for debugging
        if (response.data.recommendations && response.data.recommendations.length > 0) {
          console.log('[useRecommendations] First 3 recommendations:', 
            response.data.recommendations.slice(0, 3)
          );
        } else {
          console.warn('[useRecommendations] Empty recommendations returned from API');
        }
        
        return response.data;
      } catch (error) {
        console.error('[useRecommendations] API Error:', error);
        if (axios.isAxiosError(error)) {
          console.error('[useRecommendations] Response data:', error.response?.data);
          console.error('[useRecommendations] Response status:', error.response?.status);
        }
        throw error;
      }
    },
    // Only enable the query if searchQuery is not empty
    enabled: !!searchQuery && searchQuery.trim().length > 0,
    staleTime: 60_000,
    retry: 1,
    refetchOnWindowFocus: false,
  });
};