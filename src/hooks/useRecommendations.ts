// hooks/useRecommendations.ts
import { useQuery } from '@tanstack/react-query';
import axios from 'axios';

export const useRecommendations = (searchQuery: string) =>
  useQuery({
    queryKey: ['recs', searchQuery],
    queryFn: async () => {
      const response = await axios({
        method: 'post',
        url: 'http://10.22.134.152:8000/recommend', // Or /v1/recommendations
        headers: {
          'X-API-Key': 'testkey123',
          'Content-Type': 'application/json',
        },
        data: {
          user_id: 'user0', // You might want to make this dynamic later
          count: 5,
          search_query: searchQuery, // ADD THIS LINE
        },
      });
      return response.data; // Returns the full response object containing recommendations array
    },
    enabled: !!searchQuery, // Only run if searchQuery is not empty
    staleTime: 60_000,
  });