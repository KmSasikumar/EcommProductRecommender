// components/RecommendationCard.tsx
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
// We'll define a simple interface matching the API response structure
interface RecommendationItem {
  item_id: string; // Comes from API
  score: number;   // Comes from API
  // Add other fields if your API returns them (e.g., name, image_url, price)
}

type Props = {
  recommendation: RecommendationItem; // Renamed prop to be clearer
  onPress: () => void; // Handler for when the card is pressed (tap/view)
  onAddToCart: () => void; // Handler for when the "Add to Cart" button is pressed
};

const RecommendationCard: React.FC<Props> = ({ recommendation, onPress, onAddToCart }) => {
  // Placeholder image - replace with actual product image URL if available from API
  const productImage = 'https://via.placeholder.com/150/0000FF/808080?Text=Product+Image';
  // Derive a name for display if not provided by API
  const displayName = recommendation.item_id; // Or use a mapping if available
  const displayPrice = `$${(recommendation.score * 100).toFixed(2)}`; // Placeholder price based on score

  return (
    <TouchableOpacity style={styles.card} onPress={onPress}> {/* Add onPress here */}
      <Image 
        source={{ uri: productImage }} 
        style={styles.image}
        resizeMode="cover"
      />
      <View style={styles.detailsContainer}>
        <Text style={styles.title} numberOfLines={2}>
          {displayName} {/* Use item_id or mapped name */}
        </Text>
        <Text style={styles.score}>Score: {recommendation.score.toFixed(3)}</Text>
        <Text style={styles.price}>{displayPrice}</Text> {/* Placeholder price */}
        <TouchableOpacity style={styles.button} onPress={onAddToCart}> {/* Add onPress here */}
            <Text style={styles.buttonText}>Add to Cart</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row', // Arrange image and details horizontally
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginVertical: 4,
    marginHorizontal: 16, // Consistent with your previous margin
    elevation: 3,
    shadowColor: '#000',
    shadowOpacity: 0.1,
    shadowRadius: 4,
  },
  image: {
    width: 60,
    height: 60,
    borderRadius: 8,
  },
  detailsContainer: {
    flex: 1,
    marginLeft: 12,
    justifyContent: 'space-between', // Space content vertically
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  score: {
    fontSize: 14,
    color: '#666',
    marginTop: 4,
  },
  price: {
     fontSize: 14,
     fontWeight: 'bold',
     color: '#007bff',
     marginTop: 2,
  },
  // Removed explanation style as it's not in the basic API response
  button: {
    backgroundColor: '#28a745', // Green color for add to cart
    borderRadius: 4,
    paddingVertical: 6,
    paddingHorizontal: 12,
    alignSelf: 'flex-start', // Align button to the left
    marginTop: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
});

export default RecommendationCard;