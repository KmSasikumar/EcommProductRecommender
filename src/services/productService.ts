// services/productService.ts
import * as SQLite from 'expo-sqlite';
import { Product } from '../models/Product';

const DATABASE_NAME = 'ecommerce.db';

// Define the structure of a row returned from the database query
interface ProductRow {
  id: string;
  name: string;
  price: number;
  category: string;
  image_urls: string; // JSON string
  tags: string;       // JSON string
}

/**
 * Fetches a single product's details from the local SQLite database by its ID.
 * @param productId The ID of the product to fetch.
 * @returns A Promise that resolves to the Product object, or null if not found.
 */
export const getProductById = async (productId: string): Promise<Product | null> => {
  if (!productId) {
    return null;
  }

  let db: SQLite.SQLiteDatabase | null = null;
  try {
    db = await SQLite.openDatabaseAsync(DATABASE_NAME);

    // Explicitly type the result
    const result: ProductRow | null = await db.getFirstAsync<ProductRow>(
      `
      SELECT id, name, price, category, image_urls, tags
      FROM products
      WHERE id = ?
      `,
      [productId]
    );

    if (!result) {
      console.log(`Product with ID '${productId}' not found in database.`);
      return null;
    }

    const product_id = result.id;
    const name = result.name;
    const price = result.price;
    const category = result.category;
    const image_urls_json = result.image_urls;
    const tags_json = result.tags;

    let imageUrls: string[] = [];
    let tags: string[] = [];

    try {
      imageUrls = image_urls_json ? JSON.parse(image_urls_json) : [];
      tags = tags_json ? JSON.parse(tags_json) : [];
    } catch (parseError) {
      console.warn(`Error parsing JSON for product ${product_id}:`, parseError);
      imageUrls = [];
      tags = [];
    }

    const product: Product = {
      id: product_id,
      name: name,
      price: price,
      imageUrls: imageUrls,
      category: category,
      tags: tags,
    };

    return product;

  } catch (error) {
    console.error(`Error fetching product '${productId}' from database:`, error);
    return null;
  } finally {
    // Optional: Close the database connection if needed
    // if (db) {
    //   await db.close();
    // }
  }
};

/**
 * Fetches all products from the local SQLite database.
 * @returns A Promise that resolves to an array of Product objects.
 */
export const getAllProducts = async (): Promise<Product[]> => {
  let db: SQLite.SQLiteDatabase | null = null;
  try {
    db = await SQLite.openDatabaseAsync(DATABASE_NAME);

    // Explicitly type the results array
    const results: ProductRow[] = await db.getAllAsync<ProductRow>(`
      SELECT id, name, price, category, image_urls, tags
      FROM products
    `);

    const allProducts: Product[] = [];
    for (const row of results) {
      const product_id = row.id;
      const name = row.name;
      const price = row.price;
      const category = row.category;
      const image_urls_json = row.image_urls;
      const tags_json = row.tags;

      let imageUrls: string[] = [];
      let tags: string[] = [];

      try {
        imageUrls = image_urls_json ? JSON.parse(image_urls_json) : [];
        tags = tags_json ? JSON.parse(tags_json) : [];
      } catch (parseError) {
        console.warn(`Error parsing JSON for product ${product_id}:`, parseError);
        imageUrls = [];
        tags = [];
      }

      allProducts.push({
        id: product_id,
        name: name,
        price: price,
        imageUrls: imageUrls,
        category: category,
        tags: tags,
      });
    }

    return allProducts;

  } catch (error) {
    console.error('Error fetching all products from database:', error);
    return [];
  } finally {
    // Optional: Close the database connection if needed
    // if (db) {
    //   await db.close();
    // }
  }
};

/**
 * Searches for products in the local SQLite database based on name, category, or tags.
 * @param query The search term.
 * @returns A Promise that resolves to an array of Product objects.
 */
export const searchProducts = async (query: string): Promise<Product[]> => {
  if (!query) {
    return await getAllProducts(); // Return all products if no query
  }

  let db: SQLite.SQLiteDatabase | null = null;
  try {
    db = await SQLite.openDatabaseAsync(DATABASE_NAME);

    const searchPattern = `%${query.toLowerCase()}%`;
    
    // Explicitly type the results array
    const results: ProductRow[] = await db.getAllAsync<ProductRow>(`
      SELECT id, name, price, category, image_urls, tags
      FROM products
      WHERE LOWER(name) LIKE ? OR LOWER(category) LIKE ? OR LOWER(tags) LIKE ?
    `, [searchPattern, searchPattern, searchPattern]);

    const searchResults: Product[] = [];
    for (const row of results) {
      const product_id = row.id;
      const name = row.name;
      const price = row.price;
      const category = row.category;
      const image_urls_json = row.image_urls;
      const tags_json = row.tags;

      let imageUrls: string[] = [];
      let tags: string[] = [];

      try {
        imageUrls = image_urls_json ? JSON.parse(image_urls_json) : [];
        tags = tags_json ? JSON.parse(tags_json) : [];
      } catch (parseError) {
        console.warn(`Error parsing JSON for product ${product_id}:`, parseError);
        imageUrls = [];
        tags = [];
      }

      searchResults.push({
        id: product_id,
        name: name,
        price: price,
        imageUrls: imageUrls,
        category: category,
        tags: tags,
      });
    }

    return searchResults;

  } catch (error) {
    console.error(`Error searching products in database for query '${query}':`, error);
    return [];
  } finally {
    // Optional: Close the database connection if needed
    // if (db) {
    //   await db.close();
    // }
  }
};