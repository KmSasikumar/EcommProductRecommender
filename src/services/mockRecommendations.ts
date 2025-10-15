import type { Recommendation } from '../models';

export async function mockRecommendations(searchQuery: string): Promise<Recommendation[]> {
  await new Promise((r) => setTimeout(r, 500)); // fake delay

  const dummy: Recommendation[] = [
    {
      product: {
        id: '3',
        name: 'Running Shorts',
        price: 25.0,
        category: 'apparel',
        tags: ['sport'],
        imageUrls: [],
      },
      score: 0.87,
      explanation: `Customers who searched "${searchQuery}" also viewed Running Shorts.`,
    },
    {
      product: {
        id: '4',
        name: 'Water Bottle',
        price: 18.0,
        category: 'accessories',
        tags: ['fitness'],
        imageUrls: [],
      },
      score: 0.82,
      explanation: `Great match for "${searchQuery}" lovers.`,
    },
  ];

  return dummy;
}