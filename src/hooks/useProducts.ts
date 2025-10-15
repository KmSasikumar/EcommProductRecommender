// hooks/useProducts.ts
import { useQuery } from '@tanstack/react-query';
import { getAllProducts } from '../services/mockProducts'; // Or your real API service

export const useProducts = () => {
  return useQuery({
    queryKey: ['products'],
    queryFn: getAllProducts, // Or your real API function
    staleTime: 5 * 60 * 1000, // 5 minutes
  });
};