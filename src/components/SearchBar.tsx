// components/SearchBar.tsx
import React, { useState, useRef, forwardRef, useImperativeHandle, useEffect } from 'react';
import {
  View,
  TextInput,
  StyleSheet,
  TouchableOpacity,
  Text,
  NativeSyntheticEvent,
  TextInputSubmitEditingEventData,
  TextInput as RNTextInput,
} from 'react-native';

type Props = {
  value: string;
  onChange: (v: string) => void;
  onSubmitEditing?: () => void;
  loading?: boolean;
  placeholder?: string;
};

export type SearchBarHandle = {
  focus: () => void;
  blur: () => void;
};

const SearchBar = forwardRef<SearchBarHandle, Props>(
  ({ value, onChange, onSubmitEditing, loading = false, placeholder = 'Search…' }, ref) => {
    const [isFocused, setIsFocused] = useState(false);
    const textInputRef = useRef<RNTextInput | null>(null);

    useImperativeHandle(
      ref,
      () => ({
        focus: () => {
          console.log('[SearchBar] imperative focus() called');
          textInputRef.current?.focus();
        },
        blur: () => {
          console.log('[SearchBar] imperative blur() called');
          textInputRef.current?.blur();
        },
      }),
      []
    );

    // MOUNT / UNMOUNT logging
    useEffect(() => {
      console.log('[SearchBar] mounted');
      return () => {
        console.log('[SearchBar] unmounted');
      };
    }, []);

    const handleClear = () => {
      console.log('[SearchBar] handleClear called');
      onChange('');
      // keep focus after clear
      textInputRef.current?.focus();
    };

    const handleNativeSubmit = (e?: NativeSyntheticEvent<TextInputSubmitEditingEventData>) => {
      console.log('[SearchBar] handleNativeSubmit called - nativeEvent:', e?.nativeEvent);
      onSubmitEditing?.();
      // Do not blur here. Parent may focus back if needed.
    };

    const handleChange = (text: string) => {
      console.log('[SearchBar] onChangeText ->', text);
      onChange(text);
    };

    return (
      <View style={[styles.container, isFocused && styles.containerFocused]}>
        <TextInput
          ref={textInputRef}
          style={styles.input}
          placeholder={placeholder}
          value={value}
          onChangeText={handleChange}
          onSubmitEditing={handleNativeSubmit}
          returnKeyType="search"
          onFocus={() => {
            console.log('[SearchBar] onFocus');
            setIsFocused(true);
          }}
          onBlur={() => {
            console.log('[SearchBar] onBlur');
            setIsFocused(false);
          }}
          blurOnSubmit={false}
        />
        {value.length > 0 && !loading && (
          <TouchableOpacity style={styles.clearButton} onPress={handleClear}>
            <Text style={styles.clearButtonText}>✕</Text>
          </TouchableOpacity>
        )}
        {loading && (
          <View style={styles.loadingContainer}>
            <Text style={styles.loadingText}>🔍</Text>
          </View>
        )}
      </View>
    );
  }
);

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    height: 40,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    paddingHorizontal: 10,
    margin: 16,
    backgroundColor: '#fff',
  },
  containerFocused: {
    borderColor: '#007bff',
  },
  input: {
    flex: 1,
    height: '100%',
    fontSize: 16,
  },
  clearButton: {
    paddingHorizontal: 8,
  },
  clearButtonText: {
    fontSize: 18,
    color: '#999',
  },
  loadingContainer: {
    paddingHorizontal: 8,
  },
  loadingText: {
    fontSize: 16,
  },
});

export default SearchBar;
