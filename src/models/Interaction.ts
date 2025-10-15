export type InteractionType = 'view' | 'cart' | 'purchase';

export interface Interaction {
  userId: string;
  productId: string;
  type: InteractionType;
  ts: number; // timestamp (seconds since epoch)
}