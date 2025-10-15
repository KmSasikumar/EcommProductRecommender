// components/SearchBar.tsx
import React, { useState } from 'react';
import { View, TextInput, StyleSheet, TouchableOpacity, Text } from 'react-native';

type Props = {
  value: string;
  onChange: (v: string) => void;
  onSubmitEditing?: () => void; // Make onSubmitEditing optional
  loading?: boolean; // Add a loading prop
};

const SearchBar: React.FC<Props> = ({ value, onChange, onSubmitEditing, loading = false }) => {
  const [isFocused, setIsFocused] = useState(false); // Track focus state

  const handleClear = () => {
    onChange(''); // Clear the input value
    // Optionally, you could also call onSubmitEditing here if clearing should trigger a "show all products" action
    // onSubmitEditing?.();
  };

  return (
    <View style={[styles.container, isFocused && styles.containerFocused]}>
      <TextInput
        style={styles.input}
        placeholder="Search…"
        value={value}
        onChangeText={onChange}
        onSubmitEditing={onSubmitEditing}
        returnKeyType="search" // Shows a search icon on the keyboard
        onFocus={() => setIsFocused(true)}
        onBlur={() => setIsFocused(false)}
      />
      {/* Show Clear Button if there's text and it's not loading */}
      {value.length > 0 && !loading && (
        <TouchableOpacity style={styles.clearButton} onPress={handleClear}>
          <Text style={styles.clearButtonText}>✕</Text>
        </TouchableOpacity>
      )}
      {/* Show Loading Indicator if loading prop is true */}
      {loading && (
        <View style={styles.loadingContainer}>
          <Text style={styles.loadingText}>🔍</Text> {/* Or use an actual spinner component */}
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row', // Align input and buttons horizontally
    alignItems: 'center', // Vertically center items
    height: 40,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 10,
    margin: 16,
    backgroundColor: '#fff', // Ensure background for contrast
  },
  containerFocused: {
    borderColor: '#007bff', // Change border color when focused
  },
  input: {
    flex: 1, // Take up remaining space
    height: '100%',
    fontSize: 16,
  },
  clearButton: {
    paddingHorizontal: 8, // Add padding for easier touch
  },
  clearButtonText: {
    fontSize: 18,
    color: '#999', // Light grey color for the clear 'X'
  },
  loadingContainer: {
    paddingHorizontal: 8,
  },
  loadingText: {
    fontSize: 16,
  },
});

export default SearchBar;