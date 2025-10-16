// screens/HomeScreen.tsx
import React, { useCallback } from 'react';
import {
  View,
  Text,
  FlatList,
  StyleSheet,
  ActivityIndicator,
  TouchableOpacity,
  Image,
  GestureResponderEvent,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import { useProducts } from '../hooks/useProducts';
import { Product } from '../models/Product';

type HomeScreenNavigationProp = StackNavigationProp<RootStackParamList, 'Home'>;

const API_BASE_URL = 'http://10.22.134.152:8000';

/**
 * Local StateCard component used for Error / Empty states.
 * Uses the local asset at ../assets/alert.png for visual parity with your design.
 */
function StateCard({
  variant = 'error',
  title,
  message,
  buttonText = 'Retry',
  onRetry,
}: {
  variant?: 'error' | 'empty';
  title?: string;
  message?: string;
  buttonText?: string;
  onRetry?: (e?: GestureResponderEvent) => void;
}) {
  // Load the downloaded image asset
  const imageSource = require('../assets/alert.png');

  return (
    <View style={styles.stateWrapper}>
      <View style={styles.stateCard}>
        <View style={styles.stateImageContainer}>
          <Image source={imageSource} style={styles.stateImage} resizeMode="contain" />
        </View>

        <Text style={styles.stateTitle}>
          {title ?? (variant === 'error' ? 'Failed to load content' : 'No results found')}
        </Text>

        <Text style={styles.stateMessage}>
          {message ??
            (variant === 'error'
              ? 'Something went wrong while loading the search results. Please check your connection and try again.'
              : 'We couldn’t find any products that match your search. Try a different keyword.')}
        </Text>

        <TouchableOpacity
          style={styles.stateButton}
          onPress={onRetry}
          accessibilityRole="button"
          accessibilityLabel={buttonText}
        >
          <Text style={styles.stateButtonText}>{buttonText}</Text>
        </TouchableOpacity>
      </View>
    </View>
  );
}

export default function HomeScreen() {
  const navigation = useNavigation<HomeScreenNavigationProp>();

  const { data: products, isLoading, error, refetch } = useProducts();

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
    } catch (err) {
      console.error('Error sending interaction to backend:', err);
    }
  };

  // --- Interaction Handlers ---
  const handleProductPress = useCallback((productId: string) => {
    console.log(`Product tapped: ${productId}`);
    sendInteractionToBackend('user0', productId, 'tap');
  }, []);

  const handleAddToCart = useCallback((productId: string) => {
    console.log(`Added to cart: ${productId}`);
    sendInteractionToBackend('user0', productId, 'cart');
  }, []);

  // Navigate to search page when search bar is tapped
  const handleSearchPress = () => {
    navigation.navigate('SearchPage');
  };

  // Render Product Card in Grid Layout
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

  // --- Loading state ---
  if (isLoading) {
    return (
      <View style={styles.centerContainer}>
        <ActivityIndicator size="large" color="#007AFF" />
        <Text style={styles.loadingText}>Loading products...</Text>
      </View>
    );
  }

  // Prepare flags for empty / error handling
  const hasProducts = Array.isArray(products) && products.length > 0;
  const isError = !!error;

  // --- Main render ---
  return (
    <View style={styles.container}>
      {/* Search Bar */}
      <TouchableOpacity
        style={styles.searchBar}
        onPress={handleSearchPress}
        activeOpacity={0.7}
      >
        <Text style={styles.searchIcon}>🔍</Text>
        <Text style={styles.searchPlaceholder}>Search for products...</Text>
      </TouchableOpacity>

      {/* Subtitle */}
      <Text style={styles.subtitle}>✨ Discover amazing products</Text>

      {/* Featured Products Header */}
      <View style={styles.sectionHeader}>
        <Text style={styles.sectionTitle}>Featured Products</Text>
        <TouchableOpacity>
          <Text style={styles.viewAllText}>View All →</Text>
        </TouchableOpacity>
      </View>

      {/* If there was an error, show the error card with Retry */}
      {isError && (
        <View style={styles.emptyWrapper}>
          <StateCard
            variant="error"
            title="Failed to load content"
            message="Something went wrong while loading the search results. Please check your connection and try again."
            buttonText="Retry"
            onRetry={() => {
              refetch?.();
            }}
          />
        </View>
      )}

      {/* If no products and no error, show empty-state */}
      {!isError && !hasProducts && (
        <View style={styles.emptyWrapper}>
          <StateCard
            variant="empty"
            title="No products found"
            message="We couldn’t find any products at the moment. Try changing the search or retry."
            buttonText="Retry"
            onRetry={() => {
              refetch?.();
            }}
          />
        </View>
      )}

      {/* Products Grid */}
      {!isError && hasProducts && (
        <FlatList
          data={products || []}
          keyExtractor={(item) => item.id}
          renderItem={renderProductCard}
          numColumns={2}
          columnWrapperStyle={styles.row}
          contentContainerStyle={styles.listContainer}
          showsVerticalScrollIndicator={false}
        />
      )}

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem}>
          <Text style={[styles.navIcon, styles.navIconActive]}>🏠</Text>
          <Text style={[styles.navLabel, styles.navLabelActive]}>Home</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem} onPress={handleSearchPress}>
          <Text style={styles.navIcon}>🔍</Text>
          <Text style={styles.navLabel}>Search</Text>
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
  /* Layout */
  container: {
    flex: 1,
    backgroundColor: '#F5F5F5',
  },
  centerContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#F5F5F5',
  },

  /* State card */
  emptyWrapper: {
    paddingHorizontal: 16,
    paddingTop: 24,
    alignItems: 'center',
  },
  stateWrapper: {
    paddingHorizontal: 8,
    width: '100%',
    alignItems: 'center',
  },
  stateCard: {
    width: '100%',
    maxWidth: 520,
    backgroundColor: '#fff',
    borderRadius: 14,
    padding: 20,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOpacity: 0.08,
    shadowRadius: 12,
    elevation: 4,
  },
  stateImageContainer: {
    width: 84,
    height: 84,
    borderRadius: 42,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  stateImage: {
    width: 72,
    height: 72,
  },
  stateTitle: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 6,
    textAlign: 'center',
  },
  stateMessage: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 14,
    paddingHorizontal: 6,
  },
  stateButton: {
    backgroundColor: '#007bff',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
    minWidth: 120,
    alignItems: 'center',
  },
  stateButtonText: {
    color: '#fff',
    fontWeight: '600',
  },

  /* Rest of original styles */
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 16,
    textAlign: 'center',
    paddingHorizontal: 20,
  },
  searchBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#FFF',
    marginHorizontal: 16,
    marginTop: 16,
    marginBottom: 8,
    paddingHorizontal: 16,
    paddingVertical: 12,
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  searchIcon: {
    fontSize: 18,
    marginRight: 8,
  },
  searchPlaceholder: {
    flex: 1,
    fontSize: 16,
    color: '#999',
  },
  subtitle: {
    fontSize: 14,
    color: '#666',
    marginHorizontal: 16,
    marginBottom: 16,
  },
  sectionHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginHorizontal: 16,
    marginBottom: 12,
  },
  sectionTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#000',
  },
  viewAllText: {
    fontSize: 14,
    color: '#007AFF',
    fontWeight: '600',
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
