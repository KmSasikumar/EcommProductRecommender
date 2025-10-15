import * as functions from "firebase-functions/v1";
import axios from "axios";

interface Interaction {
  productId: string;
  type: string;
}

interface Prediction {
  product_id: string;
  name?: string;
  price?: number;
  category?: string;
  tags?: string[];
  imageUrls?: string[];
  score: number;
  explanation: string;
}

const API_URL = process.env.API_URL || "http://localhost:8000/recommend";

export const getRecommendations = functions.https.onCall(
  async (data) => {
    const interactions: Interaction[] = []; // Explicitly define the type here

    const body = {
      user_id: "user123",
      search_query: data.searchQuery || "",
      interactions: interactions.map((i) => ({
        product_id: i.productId,
        type: i.type,
      })),
    };

    const { data: preds } = await axios.post(API_URL, body);

    return preds.map((p: Prediction) => ({
      product: {
        id: p.product_id,
        name: p.name || `Product ${p.product_id}`,
        price: p.price || 0,
        category: p.category || "",
        tags: p.tags || [],
        imageUrls: p.imageUrls || [],
      },
      score: p.score,
      explanation: p.explanation,
    }));
  }
);