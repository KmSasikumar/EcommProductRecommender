// components/ProductCard.tsx
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image } from 'react-native';
import { Product } from '../models/Product';

type Props = {
  product: Product;
  onPress: () => void;
  onAddToCart: () => void;
};

const ProductCard: React.FC<Props> = ({ product, onPress, onAddToCart }) => {
  const productImage = product.imageUrls?.[0] || `https://via.placeholder.com/150/0000FF/808080?Text=${encodeURIComponent(product.name)}`;

  return (
    <TouchableOpacity style={styles.card} onPress={onPress}>
      <Image 
        source={{ uri: productImage }} 
        style={styles.image}
        resizeMode="cover"
      />
      <View style={styles.detailsContainer}>
        <Text style={styles.title} numberOfLines={2}>
          {product.name}
        </Text>
        <Text style={styles.category}>{product.category}</Text>
        <Text style={styles.price}>${product.price.toFixed(2)}</Text>
        <TouchableOpacity style={styles.button} onPress={onAddToCart}>
            <Text style={styles.buttonText}>Add to Cart</Text>
        </TouchableOpacity>
      </View>
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  card: {
    flexDirection: 'row',
    backgroundColor: '#fff',
    borderRadius: 8,
    padding: 12,
    marginVertical: 4,
    marginHorizontal: 16,
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
    justifyContent: 'center',
  },
  title: {
    fontSize: 16,
    fontWeight: 'bold',
  },
  category: {
    fontSize: 12,
    color: '#888',
    marginTop: 2,
  },
  price: {
    fontSize: 14,
    fontWeight: 'bold',
    color: '#007bff',
    marginTop: 4,
  },
  button: {
    backgroundColor: '#28a745',
    borderRadius: 4,
    paddingVertical: 6,
    paddingHorizontal: 12,
    alignSelf: 'flex-start',
    marginTop: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 12,
    fontWeight: 'bold',
  },
});

export default ProductCard;