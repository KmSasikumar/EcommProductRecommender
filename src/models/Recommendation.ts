import type { Product } from './Product';

export interface Recommendation {
  product: Product;
  score: number; // 0..1
  explanation: string; // LLM text
}