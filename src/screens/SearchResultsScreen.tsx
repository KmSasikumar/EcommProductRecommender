// screens/SearchResultsScreen.tsx
import React, { useState, useCallback, useMemo, useRef, useEffect } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Image,
} from 'react-native';
import { useRoute, RouteProp, useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import SearchBar from '../components/SearchBar';
import type { SearchBarHandle } from '../components/SearchBar';
import { useRecommendations } from '../hooks/useRecommendations';
import { useSearchProducts } from '../hooks/useSearchProducts';
import { Product } from '../models/Product';

type SearchResultsScreenRouteProp = RouteProp<RootStackParamList, 'SearchResults'>;
type SearchResultsScreenNavigationProp = StackNavigationProp<
  RootStackParamList,
  'SearchResults'
>;

// Define the shape of a recommendation item from the NCF API
interface RecommendationItemFromAPI {
  item_id: string;
  score: number;
}

const API_BASE_URL = 'http://10.22.134.152:8000';

export default function SearchResultsScreen() {
  const route = useRoute<SearchResultsScreenRouteProp>();
  const navigation = useNavigation<SearchResultsScreenNavigationProp>();
  const searchInputRef = useRef<SearchBarHandle>(null);

  const [rawQuery, setRawQuery] = useState(route.params?.initialQuery || '');
  const [submittedQuery, setSubmittedQuery] = useState(route.params?.initialQuery || '');
  const [isShowingNCF, setIsShowingNCF] = useState(!!route.params?.initialQuery);

  // --- Fetch NCF recommendations when Enter is pressed ---
  const {
    data: recommendationsData,
    isLoading: recsLoading,
    error: recsError,
  } = useRecommendations(submittedQuery);

  // --- Fetch live search results as user types ---
  const {
    data: searchResultsData,
    isLoading: searchResultsLoading,
    error: searchResultsError,
  } = useSearchProducts(rawQuery);

  // --- Map NCF Recommendations to Product Objects ---
  const ncfRecommendations: Product[] = useMemo(() => {
    console.log('Recommendations Data:', recommendationsData); // Debug log
    
    if (!recommendationsData?.recommendations || recommendationsData.recommendations.length === 0) {
      console.log('No recommendations returned from API');
      return [];
    }

    console.log('Mapping recommendations:', recommendationsData.recommendations);
    
    // Map item_id to actual product details from database
    const productMapping: { [key: string]: { name: string; price: number; category: string; tags: string[] } } = {
      'item0': { name: 'Gaming Laptop', price: 1299.99, category: 'Electronics', tags: ['gaming', 'laptop', 'high-performance'] },
      'item1': { name: 'Wireless Bluetooth Earbuds', price: 129.99, category: 'Electronics', tags: ['audio', 'wireless', 'bluetooth'] },
      'item2': { name: 'Smart Fitness Watch', price: 199.95, category: 'Wearables', tags: ['fitness', 'smartwatch', 'health'] },
      'item3': { name: 'Ergonomic Office Chair', price: 249.50, category: 'Furniture', tags: ['office', 'ergonomic', 'comfort'] },
      'item4': { name: 'Mechanical Gaming Keyboard', price: 89.99, category: 'Electronics', tags: ['gaming', 'keyboard', 'mechanical'] },
    };
    
    return recommendationsData.recommendations.map((rec: RecommendationItemFromAPI) => {
      const productInfo = productMapping[rec.item_id];
      
      if (productInfo) {
        return {
          id: rec.item_id,
          name: productInfo.name,
          price: productInfo.price,
          imageUrls: [`https://via.placeholder.com/150/0000FF/808080?Text=${encodeURIComponent(productInfo.name)}`],
          category: productInfo.category,
          tags: [...productInfo.tags, `NCF Score: ${rec.score.toFixed(3)}`],
        };
      } else {
        // Fallback for unknown items
        return {
          id: rec.item_id,
          name: rec.item_id,
          price: rec.score * 100,
          imageUrls: [],
          category: 'Recommended',
          tags: [`NCF Score: ${rec.score.toFixed(3)}`],
        };
      }
    });
  }, [recommendationsData]);

  // --- Initialize with route params ---
  useEffect(() => {
    if (route.params?.initialQuery) {
      console.log('Initial query from route params:', route.params.initialQuery);
      setRawQuery(route.params.initialQuery);
      setSubmittedQuery(route.params.initialQuery);
      setIsShowingNCF(true);
    }
  }, [route.params?.initialQuery]);

  // Log when submittedQuery changes
  useEffect(() => {
    console.log('Submitted query changed to:', submittedQuery);
  }, [submittedQuery]);

  // --- Function to send interaction to backend ---
  const sendInteractionToBackend = async (
    userId: string,
    itemId: string,
    type: 'tap' | 'cart'
  ) => {
    try {
      const response = await fetch(`${API_BASE_URL}/interactions`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          user_id: userId,
          item_id: itemId,
          type: type,
        }),
      });

      if (!response.ok) {
        console.error(
          `Failed to send interaction to backend: ${response.status} ${response.statusText}`
        );
      } else {
        console.log(`Interaction sent successfully: ${type} on ${itemId}`);
      }
    } catch (error) {
      console.error('Error sending interaction to backend:', error);
    }
  };

  // --- Handlers ---
  const handleSearchChange = (text: string) => {
    console.log('Search text changed:', text);
    setRawQuery(text);

    // If user clears the search, reset NCF state
    if (!text.trim()) {
      setSubmittedQuery('');
      setIsShowingNCF(false);
    } else {
      // When typing, switch to live search mode
      setIsShowingNCF(false);
    }
  };

  const handleSearchSubmit = () => {
    const trimmedQuery = rawQuery.trim();
    console.log('NCF Search submitted with query:', trimmedQuery);

    if (trimmedQuery) {
      // Trigger NCF recommendations when Enter is pressed
      setSubmittedQuery(trimmedQuery);
      setIsShowingNCF(true);
    } else {
      // If query is empty on submit, clear NCF state
      setSubmittedQuery('');
      setIsShowingNCF(false);
    }

    // Keep keyboard open by refocusing
    setTimeout(() => {
      searchInputRef.current?.focus();
    }, 100);
  };

  const handleProductPress = useCallback((productId: string) => {
    console.log(`Product tapped: ${productId}`);
    sendInteractionToBackend('user0', productId, 'tap');
  }, []);

  const handleAddToCart = useCallback((productId: string) => {
    console.log(`Added to cart: ${productId}`);
    sendInteractionToBackend('user0', productId, 'cart');
  }, []);

  const handleHomePress = () => {
    navigation.navigate('Home');
  };

  const handleSearchPress = () => {
    navigation.goBack();
  };

  // --- Render Product Card in Grid Layout ---
  const renderProductCard = ({ item }: { item: Product }) => {
    const productImage =
      item.imageUrls?.[0] ||
      `https://via.placeholder.com/150/0000FF/808080?Text=${encodeURIComponent(item.name)}`;

    return (
      <TouchableOpacity
        style={styles.productCard}
        onPress={() => handleProductPress(item.id)}
      >
        <Image
          source={{ uri: productImage }}
          style={styles.productImage}
          resizeMode="cover"
        />
        <View style={styles.productDetails}>
          <Text style={styles.productName} numberOfLines={2}>
            {item.name}
          </Text>
          <View style={styles.priceRatingRow}>
            <Text style={styles.productPrice}>${item.price.toFixed(2)}</Text>
            <View style={styles.ratingContainer}>
              <Text style={styles.starIcon}>⭐</Text>
              <Text style={styles.ratingText}>4.8</Text>
            </View>
          </View>
          <View style={styles.tagsContainer}>
            {item.tags?.slice(0, 2).map((tag, index) => (
              <View key={index} style={styles.tag}>
                <Text style={styles.tagText}>{tag}</Text>
              </View>
            ))}
          </View>
          <TouchableOpacity
            style={styles.addToCartButton}
            onPress={() => handleAddToCart(item.id)}
          >
            <Text style={styles.addToCartText}>🛒 Add to Cart</Text>
          </TouchableOpacity>
        </View>
      </TouchableOpacity>
    );
  };

  // --- Determine Loading and Error States ---
  const isLoading = isShowingNCF ? recsLoading : searchResultsLoading;
  const error = isShowingNCF ? recsError : searchResultsError;

  // --- Determine Display Data ---
  const displayData = isShowingNCF ? ncfRecommendations : searchResultsData;

  console.log('Current state:', {
    isShowingNCF,
    submittedQuery,
    rawQuery,
    ncfRecommendationsCount: ncfRecommendations?.length || 0,
    searchResultsCount: searchResultsData?.length || 0,
    displayDataCount: displayData?.length || 0,
  });

  // --- Status Text ---
  const statusText = isShowingNCF
    ? 'Showing: NCF Recommendations'
    : rawQuery.trim()
    ? 'Showing: Live search results'
    : 'Type and press Enter for recommendations';

  return (
    <View style={styles.container}>
      {/* Search Bar */}
      <SearchBar
        ref={searchInputRef}
        value={rawQuery}
        onChange={handleSearchChange}
        onSubmitEditing={handleSearchSubmit}
        loading={isLoading}
        placeholder="Type and press Enter for recommendations..."
      />

      {/* Status Text */}
      <View style={styles.statusContainer}>
        <Text style={styles.statusText}>✨ {statusText}</Text>
      </View>

      {/* Loading State */}
      {isLoading && (
        <View style={styles.centerContainer}>
          <ActivityIndicator size="large" color="#007AFF" />
          <Text style={styles.loadingText}>
            {isShowingNCF ? 'Getting Recommendations...' : 'Searching Products...'}
          </Text>
        </View>
      )}

      {/* Error State */}
      {error && (
        <View style={styles.centerContainer}>
          <Text style={styles.errorText}>Error: {error.message}</Text>
          <Text
            style={styles.retryText}
            onPress={() => {
              setSubmittedQuery('');
              setRawQuery('');
              setIsShowingNCF(false);
            }}
          >
            Tap to retry
          </Text>
        </View>
      )}

      {/* Products Grid */}
      {!isLoading && !error && (
        <FlatList
          data={displayData || []}
          keyExtractor={(item) => item.id}
          renderItem={renderProductCard}
          numColumns={2}
          columnWrapperStyle={styles.row}
          contentContainerStyle={styles.listContainer}
          ListEmptyComponent={
            <View style={styles.emptyContainer}>
              <Text style={styles.emptyText}>
                {isShowingNCF
                  ? `No recommendations found for "${submittedQuery}"`
                  : rawQuery
                  ? `No products found for "${rawQuery}"`
                  : 'Start typing to search for products or press Enter for recommendations'}
              </Text>
              {isShowingNCF && (
                <Text style={styles.emptySubtext}>
                  The API returned empty recommendations. Please check:
                  {'\n'}• If the query is valid
                  {'\n'}• If there are interactions in the database
                  {'\n'}• If the NCF model is trained
                </Text>
              )}
            </View>
          }
          showsVerticalScrollIndicator={false}
          keyboardShouldPersistTaps="handled"
        />
      )}

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem} onPress={handleHomePress}>
          <Text style={styles.navIcon}>🏠</Text>
          <Text style={styles.navLabel}>Home</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={handleSearchPress}>
          <Text style={[styles.navIcon, styles.navIconActive]}>🔍</Text>
          <Text style={[styles.navLabel, styles.navLabelActive]}>Search</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <View style={styles.cartBadgeContainer}>
            <Text style={styles.navIcon}>🛒</Text>
            <View style={styles.cartBadge}>
              <Text style={styles.cartBadgeText}>3</Text>
            </View>
          </View>
          <Text style={styles.navLabel}>Cart</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
          <Text style={styles.navIcon}>👤</Text>
          <Text style={styles.navLabel}>Profile</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  statusContainer: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#FFF',
  },
  statusText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 20,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    color: '#FF3B30',
    marginBottom: 10,
    fontSize: 16,
    textAlign: 'center',
  },
  retryText: {
    color: '#007AFF',
    textDecorationLine: 'underline',
    fontSize: 16,
  },
  listContainer: {
    paddingHorizontal: 8,
    paddingBottom: 100,
  },
  row: {
    justifyContent: 'space-between',
    paddingHorizontal: 8,
  },
  productCard: {
    flex: 1,
    backgroundColor: '#FFF',
    borderRadius: 16,
    margin: 8,
    maxWidth: '47%',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
    overflow: 'hidden',
  },
  productImage: {
    width: '100%',
    height: 140,
    backgroundColor: '#E0E0E0',
  },
  productDetails: {
    padding: 12,
  },
  productName: {
    fontSize: 14,
    fontWeight: '600',
    color: '#000',
    marginBottom: 6,
    minHeight: 36,
  },
  priceRatingRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  productPrice: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#007AFF',
  },
  ratingContainer: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  starIcon: {
    fontSize: 14,
    marginRight: 2,
  },
  ratingText: {
    fontSize: 12,
    color: '#666',
    fontWeight: '600',
  },
  tagsContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginBottom: 8,
  },
  tag: {
    backgroundColor: '#F0F0F0',
    paddingHorizontal: 8,
    paddingVertical: 4,
    borderRadius: 6,
    marginRight: 4,
    marginBottom: 4,
  },
  tagText: {
    fontSize: 10,
    color: '#666',
  },
  addToCartButton: {
    backgroundColor: '#007AFF',
    paddingVertical: 10,
    borderRadius: 8,
    alignItems: 'center',
  },
  addToCartText: {
    color: '#FFF',
    fontSize: 12,
    fontWeight: 'bold',
  },
  emptyContainer: {
    padding: 40,
    alignItems: 'center',
  },
  emptyText: {
    fontSize: 16,
    color: '#666',
    textAlign: 'center',
    marginBottom: 8,
  },
  emptySubtext: {
    fontSize: 13,
    color: '#999',
    textAlign: 'center',
    marginTop: 8,
  },
  bottomNav: {
    position: 'absolute',
    bottom: 0,
    left: 0,
    right: 0,
    flexDirection: 'row',
    backgroundColor: '#FFF',
    paddingVertical: 12,
    paddingBottom: 20,
    borderTopWidth: 1,
    borderTopColor: '#E0E0E0',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: -2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 8,
  },
  navItem: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  navIcon: {
    fontSize: 24,
    marginBottom: 4,
    opacity: 0.6,
  },
  navIconActive: {
    opacity: 1,
  },
  navLabel: {
    fontSize: 11,
    color: '#666',
  },
  navLabelActive: {
    color: '#007AFF',
    fontWeight: '600',
  },
  cartBadgeContainer: {
    position: 'relative',
  },
  cartBadge: {
    position: 'absolute',
    top: -4,
    right: -8,
    backgroundColor: '#FF3B30',
    borderRadius: 10,
    width: 18,
    height: 18,
    justifyContent: 'center',
    alignItems: 'center',
  },
  cartBadgeText: {
    color: '#FFF',
    fontSize: 10,
    fontWeight: 'bold',
  },
});