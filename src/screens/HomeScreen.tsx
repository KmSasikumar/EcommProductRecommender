// screens/HomeScreen.tsx
import React, { useState, useCallback, useEffect } from 'react';
import { View, FlatList, StyleSheet, ActivityIndicator, Text } from 'react-native';
import SearchBar from '../components/SearchBar';
import ProductCard from '../components/ProductCard';
// import { useProducts } from '../hooks/useProducts'; // Potentially remove if useSearchProducts handles initial load
import { useRecommendations } from '../hooks/useRecommendations';
import { useSearchProducts } from '../hooks/useSearchProducts'; // Use the updated hook
import { Product } from '../models/Product';

// Define interaction types matching the Python enum
type InteractionType = 'tap' | 'cart';

// Define the structure matching the API's RecommendationItem (from NCF)
interface RecommendationItemFromAPI {
  item_id: string;
  score: number;
}

// Define the structure for the interaction request payload
interface InteractionPayload {
  user_id: string;
  item_id: string;
  type: InteractionType;
  timestamp?: number; // Optional, client can send, server will default if missing
}

export default function HomeScreen() {
  const [rawQuery, setRawQuery] = useState(''); // State for the immediate input (what's shown in the search bar)
  const [searchQuery, setSearchQuery] = useState(''); // State that triggers the NCF API call (updated only on Enter)
  const [isSearchingNCF, setIsSearchingNCF] = useState(false); // Track if we are currently in NCF search state (Enter pressed)

  // --- Hooks for Data Fetching ---

  // Use 'rawQuery' for live search results (including initial load when rawQuery is '')
  // This hook now calls the database-backed /search endpoint
  const { data: searchResultsData, isLoading: searchResultsLoading, error: searchResultsError } = useSearchProducts(rawQuery);

  // Use 'searchQuery' for NCF recommendations (triggered on Enter)
  const { data: recommendationsData, isLoading: recsLoading, error: recsError } = useRecommendations(searchQuery);

  // --- Determine Loading and Error States ---
  const isLoading = isSearchingNCF ? recsLoading : searchResultsLoading;
  const error = isSearchingNCF ? recsError : searchResultsError;

  // --- Map NCF Recommendations to Product Objects ---
  const ncfRecommendations: Product[] = recommendationsData?.recommendations.map((rec: RecommendationItemFromAPI) => ({
    id: rec.item_id,
    name: rec.item_id, // Or use a mapping if available
    price: rec.score * 100, // Placeholder price based on score
    imageUrls: [], // Placeholder image handled in ProductCard
    category: 'Recommended',
    tags: [`Score: ${rec.score.toFixed(3)}`], // Include score as a tag
  })) || [];

  // --- Determine Display Data ---
  // 1. If NCF search is active (isSearchingNCF is true), use NCF recommendations
  // 2. Otherwise, use data from the database-backed search hook (initial list or filtered results)
  const displayData = isSearchingNCF ? ncfRecommendations : searchResultsData;

  // --- Handlers ---

  const handleSearchChange = (text: string) => {
    setRawQuery(text); // Update the display input immediately as user types
    // Do NOT update searchQuery here, so NCF API is not called on every keystroke
    // Clear searchQuery and reset NCF search state if rawQuery becomes empty
    if (!text) {
      setSearchQuery('');
      setIsSearchingNCF(false);
    }
  };

  const handleSearchSubmit = () => {
    console.log('NCF Search submitted with query:', rawQuery);
    // Update the searchQuery state only when Enter is pressed
    // This will trigger the useRecommendations hook for NCF-based results
    setSearchQuery(rawQuery);
    setIsSearchingNCF(true); // Set flag to indicate NCF search is active
    // Optionally, clear rawQuery if desired after submitting
    // setRawQuery('');
  };

  // --- Function to send interaction to backend ---
  const sendInteractionToBackend = async (userId: string, itemId: string, type: InteractionType) => {
    try {
      const response = await fetch(`http://10.22.134.152:8000/interactions`, { // Ensure this IP/port is correct
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId, // Replace 'user0' with actual user ID if available
          item_id: itemId,
          type: type,
          // Optionally, send client-side timestamp
          // timestamp: Date.now() / 1000 // Convert milliseconds to seconds
        } as InteractionPayload),
      });

      if (!response.ok) {
        console.error(`Failed to send interaction to backend: ${response.status} ${response.statusText}`);
        // Handle error appropriately (e.g., log, retry, show message)
      } else {
        console.log(`Interaction sent successfully: ${type} on ${itemId}`);
      }
    } catch (error) {
      // Catch network errors like 'Failed to fetch' or timeouts
      console.error('Error sending interaction to backend:', error);
      // Handle error appropriately (e.g., log, show message, queue for retry later)
    }
  };

  // --- Interaction Handlers ---
  const handleProductPress = useCallback((productId: string) => {
    console.log(`Product tapped: ${productId}`);
    // Send the tap interaction to the backend
    sendInteractionToBackend('user0', productId, 'tap'); // Replace 'user0' with actual user ID
    // Add local UI feedback logic here if needed (e.g., temporary highlight)
  }, []);

  const handleAddToCart = useCallback((productId: string) => {
    console.log(`Added to cart: ${productId}`);
    // Send the cart interaction to the backend
    sendInteractionToBackend('user0', productId, 'cart'); // Replace 'user0' with actual user ID
    // Add local UI feedback logic here if needed (e.g., alert, cart badge update)
  }, []);

  // --- Render function for ProductCard ---
  const renderProductItem = ({ item }: { item: Product }) => {
    return (
      <ProductCard
        product={item}
        onPress={() => handleProductPress(item.id)} // Pass the interaction handler
        onAddToCart={() => handleAddToCart(item.id)} // Pass the interaction handler
      />
    );
  };

  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text>Loading...</Text>
      </View>
    );
  }

  if (error) {
    return (
      <View style={styles.centerContainer}>
        <Text style={styles.errorText}>Error: {error.message}</Text>
        <Text style={styles.retryText} onPress={() => window.location.reload()}>Tap to retry</Text>
      </View>
    );
  }

  // --- Main Render (FlatList) ---
  // Ensure no plain text or stray characters are direct children of the View returned by FlatList
  return (
    <FlatList
      data={displayData || []}
      keyExtractor={(item) => item.id} // Use the id field which exists on Product objects
      ListHeaderComponent={
        <SearchBar
          value={rawQuery} // Pass the immediate input value for display
          onChange={handleSearchChange}
          onSubmitEditing={handleSearchSubmit} // Call handleSearchSubmit when Enter is pressed
          loading={isLoading} // Show loading indicator based on current search type
        />
      }
      renderItem={renderProductItem} // Use the unified render function
      contentContainerStyle={[styles.listContainer, { paddingBottom: 24 }]}
    />
  );
}

const styles = StyleSheet.create({
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  errorText: {
    color: 'red',
    marginBottom: 10,
  },
  retryText: {
    color: 'blue',
    textDecorationLine: 'underline',
  },
  listContainer: {
    padding: 8,
  },
});