// screens/SearchPage.tsx
import React, { useState, useCallback, useRef } from 'react';
import {
  View,
  Text,
  StyleSheet,
  TouchableOpacity,
  FlatList,
  ActivityIndicator,
} from 'react-native';
import { useNavigation } from '@react-navigation/native';
import { StackNavigationProp } from '@react-navigation/stack';
import { RootStackParamList } from '../navigation/RootNavigator';
import SearchBar from '../components/SearchBar';
import type { SearchBarHandle } from '../components/SearchBar';
import ProductCard from '../components/ProductCard';
import { useSearchProducts } from '../hooks/useSearchProducts';
import { Product } from '../models/Product';

type SearchPageNavigationProp = StackNavigationProp<RootStackParamList, 'SearchPage'>;

const API_BASE_URL = 'http://10.22.134.152:8000';

// Categories for filter pills
const CATEGORIES = ['Electronics', 'Clothing', 'Home', 'Books'];

export default function SearchPage() {
  const navigation = useNavigation<SearchPageNavigationProp>();
  const searchInputRef = useRef<SearchBarHandle>(null);

  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('Electronics');
  const [recentSearches, setRecentSearches] = useState([
    'iPhone case',
    'Wireless headphones',
    'Running shoes',
  ]);

  // Fetch live search results
  const {
    data: searchResults,
    isLoading,
    error,
  } = useSearchProducts(searchQuery);

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
    setSearchQuery(text);
  };

  const handleSearchSubmit = () => {
    const trimmedQuery = searchQuery.trim();
    if (trimmedQuery) {
      // Add to recent searches if not already present
      if (!recentSearches.includes(trimmedQuery)) {
        setRecentSearches((prev) => [trimmedQuery, ...prev.slice(0, 4)]);
      }
      // Navigate to NCF recommendations screen
      navigation.navigate('SearchResults', { initialQuery: trimmedQuery });
    }
  };

  const handleRecentSearchPress = (query: string) => {
    setSearchQuery(query);
    searchInputRef.current?.focus();
  };

  const handleRemoveRecentSearch = (query: string) => {
    setRecentSearches((prev) => prev.filter((item) => item !== query));
  };

  const handleProductPress = useCallback((productId: string) => {
    console.log(`Product tapped: ${productId}`);
    sendInteractionToBackend('user0', productId, 'tap');
  }, []);

  const handleAddToCart = useCallback((productId: string) => {
    console.log(`Added to cart: ${productId}`);
    sendInteractionToBackend('user0', productId, 'cart');
  }, []);

  const handleBackPress = () => {
    navigation.goBack();
  };

  const handleHomePress = () => {
    navigation.navigate('Home');
  };

  // Render Product Card
  const renderProductItem = ({ item }: { item: Product }) => {
    return (
      <ProductCard
        product={item}
        onPress={() => handleProductPress(item.id)}
        onAddToCart={() => handleAddToCart(item.id)}
      />
    );
  };

  // Render Recent Search Item
  const renderRecentSearchItem = ({ item }: { item: string }) => {
    return (
      <View style={styles.recentSearchItem}>
        <TouchableOpacity
          style={styles.recentSearchContent}
          onPress={() => handleRecentSearchPress(item)}
        >
          <Text style={styles.searchIcon}>🔍</Text>
          <Text style={styles.recentSearchText}>{item}</Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={styles.removeButton}
          onPress={() => handleRemoveRecentSearch(item)}
        >
          <Text style={styles.removeButtonText}>✕</Text>
        </TouchableOpacity>
      </View>
    );
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={handleBackPress} style={styles.backButton}>
          <Text style={styles.backIcon}>←</Text>
        </TouchableOpacity>
        <Text style={styles.headerTitle}>Search Products</Text>
        <TouchableOpacity style={styles.filterButton}>
          <Text style={styles.filterIcon}>⚙️</Text>
        </TouchableOpacity>
      </View>

      {/* Search Bar */}
      <SearchBar
        ref={searchInputRef}
        value={searchQuery}
        onChange={handleSearchChange}
        onSubmitEditing={handleSearchSubmit}
        loading={isLoading}
        placeholder="Type to search products..."
      />

      {/* Category Pills */}
      <View style={styles.categoriesContainer}>
        {CATEGORIES.map((category) => (
          <TouchableOpacity
            key={category}
            style={[
              styles.categoryPill,
              selectedCategory === category && styles.categoryPillActive,
            ]}
            onPress={() => setSelectedCategory(category)}
          >
            <Text
              style={[
                styles.categoryText,
                selectedCategory === category && styles.categoryTextActive,
              ]}
            >
              {category}
            </Text>
          </TouchableOpacity>
        ))}
      </View>

      {/* Content Area */}
      {searchQuery.trim() === '' ? (
        // Show Recent Searches when no query
        <View style={styles.recentSearchesContainer}>
          <View style={styles.recentSearchesHeader}>
            <Text style={styles.clockIcon}>🕐</Text>
            <Text style={styles.recentSearchesTitle}>Recent searches</Text>
          </View>
          <FlatList
            data={recentSearches}
            keyExtractor={(item, index) => `${item}-${index}`}
            renderItem={renderRecentSearchItem}
            contentContainerStyle={styles.recentSearchesList}
            showsVerticalScrollIndicator={false}
          />
        </View>
      ) : (
        // Show Live Search Results
        <View style={styles.resultsContainer}>
          {isLoading && (
            <View style={styles.loadingContainer}>
              <ActivityIndicator size="large" color="#007AFF" />
              <Text style={styles.loadingText}>Searching products...</Text>
            </View>
          )}

          {error && (
            <View style={styles.errorContainer}>
              <Text style={styles.errorText}>Error: {error.message}</Text>
            </View>
          )}

          {!isLoading && !error && (
            <FlatList
              data={searchResults || []}
              keyExtractor={(item) => item.id}
              renderItem={renderProductItem}
              contentContainerStyle={styles.listContainer}
              ListEmptyComponent={
                <View style={styles.emptyContainer}>
                  <Text style={styles.emptyText}>
                    No products found for "{searchQuery}"
                  </Text>
                  <Text style={styles.emptySubtext}>
                    Try pressing Enter for AI recommendations
                  </Text>
                </View>
              }
              showsVerticalScrollIndicator={false}
            />
          )}
        </View>
      )}

      {/* Bottom Navigation */}
      <View style={styles.bottomNav}>
        <TouchableOpacity style={styles.navItem} onPress={handleHomePress}>
          <Text style={styles.navIcon}>🏠</Text>
          <Text style={styles.navLabel}>Home</Text>
        </TouchableOpacity>
        <TouchableOpacity style={styles.navItem}>
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
  header: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  backButton: {
    padding: 4,
  },
  backIcon: {
    fontSize: 24,
    color: '#000',
  },
  headerTitle: {
    fontSize: 18,
    fontWeight: '600',
    color: '#000',
    flex: 1,
    textAlign: 'center',
    marginHorizontal: 16,
  },
  filterButton: {
    padding: 4,
  },
  filterIcon: {
    fontSize: 20,
  },
  categoriesContainer: {
    flexDirection: 'row',
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: '#FFF',
    borderBottomWidth: 1,
    borderBottomColor: '#E0E0E0',
  },
  categoryPill: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    borderRadius: 20,
    backgroundColor: '#F0F0F0',
    marginRight: 8,
  },
  categoryPillActive: {
    backgroundColor: '#007AFF',
  },
  categoryText: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  categoryTextActive: {
    color: '#FFF',
    fontWeight: '600',
  },
  recentSearchesContainer: {
    flex: 1,
    backgroundColor: '#FFF',
    paddingTop: 16,
  },
  recentSearchesHeader: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 16,
    marginBottom: 12,
  },
  clockIcon: {
    fontSize: 18,
    marginRight: 8,
  },
  recentSearchesTitle: {
    fontSize: 14,
    color: '#666',
    fontWeight: '500',
  },
  recentSearchesList: {
    paddingHorizontal: 16,
  },
  recentSearchItem: {
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'space-between',
    paddingVertical: 12,
    borderBottomWidth: 1,
    borderBottomColor: '#F0F0F0',
  },
  recentSearchContent: {
    flex: 1,
    flexDirection: 'row',
    alignItems: 'center',
  },
  searchIcon: {
    fontSize: 18,
    marginRight: 12,
    color: '#666',
  },
  recentSearchText: {
    fontSize: 16,
    color: '#000',
  },
  removeButton: {
    padding: 8,
  },
  removeButtonText: {
    fontSize: 18,
    color: '#999',
  },
  resultsContainer: {
    flex: 1,
  },
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingTop: 40,
  },
  loadingText: {
    marginTop: 12,
    fontSize: 16,
    color: '#666',
  },
  errorContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    paddingHorizontal: 20,
  },
  errorText: {
    color: '#FF3B30',
    fontSize: 16,
    textAlign: 'center',
  },
  listContainer: {
    paddingBottom: 100,
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
    fontSize: 14,
    color: '#999',
    textAlign: 'center',
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