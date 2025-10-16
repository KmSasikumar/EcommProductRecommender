// services/localProductSearch.ts
import * as SQLite from 'expo-sqlite';
import { Product } from '../models/Product'; // Adjust the path as needed

// --- Configuration ---
const DATABASE_NAME = 'ecommerce.db'; // Name of your SQLite database file

/**
 * Searches for products directly in the local SQLite database.
 * @param queryTerm The search string provided by the user.
 * @returns A promise that resolves to an array of Product objects matching the query.
 */
export const searchProductsLocally = async (queryTerm: string): Promise<Product[]> => {
  if (!queryTerm) {
    // If no query, return an empty array or fetch all products
    // Returning all might be slow, so consider pagination or limiting results
    return [];
    // To fetch all: return getAllProductsLocally();
  }

  const db = await SQLite.openDatabaseAsync(DATABASE_NAME);

  try {
    const queryPattern = `%${queryTerm.toLowerCase()}%`;

    // Execute the query against the local database
    // This assumes your 'products' table has columns: id, name, price, category, image_urls (TEXT), tags (TEXT)
    // image_urls and tags are stored as JSON strings, so we fetch them as strings and parse them later
    const results = await db.getAllAsync(
      `
      SELECT id, name, price, category, image_urls, tags
      FROM products
      WHERE LOWER(name) LIKE ? OR LOWER(category) LIKE ? OR LOWER(tags) LIKE ?
      `,
      [queryPattern, queryPattern, queryPattern]
    );

    // Map the database rows to Product objects
    const searchResults: Product[] = results.map((row: any) => {
      let imageUrls: string[] = [];
      let tags: string[] = [];

      try {
        // Parse the JSON strings back into arrays
        imageUrls = row.image_urls ? JSON.parse(row.image_urls) : [];
        tags = row.tags ? JSON.parse(row.tags) : [];
      } catch (parseError) {
        console.warn(`Error parsing JSON for product ${row.id}:`, parseError);
        // If parsing fails, use empty arrays
        imageUrls = [];
        tags = [];
      }

      return {
        id: row.id,
        name: row.name,
        price: row.price,
        category: row.category,
        imageUrls: imageUrls,
        tags: tags,
      };
    });

    return searchResults;
  } catch (error) {
    console.error('Error searching products locally:', error);
    throw new Error(`Failed to search products locally: ${error}`);
  } finally {
    // It's generally good practice to close the database connection,
    // but expo-sqlite handles connection pooling, so closing might not be strictly necessary
    // for every single query. However, if you open many databases, closing is important.
    // await db.close(); // Uncomment if you want to explicitly close, but usually not needed for single query
  }
};

/**
 * (Optional) Fetches all products from the local database.
 * This could be used when queryTerm is empty.
 * @returns A promise that resolves to an array of all Product objects.
 */
export const getAllProductsLocally = async (): Promise<Product[]> => {
  const db = await SQLite.openDatabaseAsync(DATABASE_NAME);

  try {
    const results = await db.getAllAsync('SELECT id, name, price, category, image_urls, tags FROM products');

    const allProducts: Product[] = results.map((row: any) => {
      let imageUrls: string[] = [];
      let tags: string[] = [];

      try {
        imageUrls = row.image_urls ? JSON.parse(row.image_urls) : [];
        tags = row.tags ? JSON.parse(row.tags) : [];
      } catch (parseError) {
        console.warn(`Error parsing JSON for product ${row.id}:`, parseError);
        imageUrls = [];
        tags = [];
      }

      return {
        id: row.id,
        name: row.name,
        price: row.price,
        category: row.category,
        imageUrls: imageUrls,
        tags: tags,
      };
    });

    return allProducts;
  } catch (error) {
    console.error('Error fetching all products locally:', error);
    throw new Error(`Failed to fetch all products locally: ${error}`);
  }
};