// components/EmptyStateCard.tsx
import React from 'react';
import { View, Text, StyleSheet, TouchableOpacity, GestureResponderEvent } from 'react-native';

type Variant = 'error' | 'empty' | 'info';

type Props = {
  variant?: Variant;
  title?: string;
  message?: string;
  buttonText?: string;
  onRetry?: (e?: GestureResponderEvent) => void;
  // optional small note under the button
  note?: string;
};

const iconFor = (variant: Variant) => {
  switch (variant) {
    case 'error':
      return '⚠️';
    case 'empty':
      return '🔍';
    default:
      return 'ℹ️';
  }
};

export default function EmptyStateCard({
  variant = 'error',
  title,
  message,
  buttonText = 'Retry',
  onRetry,
  note,
}: Props) {
  return (
    <View style={styles.wrapper}>
      <View style={styles.card}>
        <View style={[styles.iconContainer, variant === 'error' ? styles.iconError : styles.iconNeutral]}>
          <Text style={styles.iconText}>{iconFor(variant)}</Text>
        </View>

        <Text style={styles.title}>{title ?? (variant === 'error' ? 'Failed to load content' : 'No results found')}</Text>
        <Text style={styles.message}>{message ?? (variant === 'error' ? 'Something went wrong while loading the search results. Please check your connection and try again.' : 'We couldn’t find any products that match your search.')}</Text>

        <TouchableOpacity
          onPress={onRetry}
          style={styles.button}
          accessibilityRole="button"
          accessibilityLabel={buttonText}
        >
          <Text style={styles.buttonText}>{buttonText}</Text>
        </TouchableOpacity>

        {note ? <Text style={styles.note}>{note}</Text> : null}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    paddingHorizontal: 16,
    paddingVertical: 24,
    alignItems: 'center',
  },
  card: {
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
  iconContainer: {
    width: 64,
    height: 64,
    borderRadius: 32,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 12,
  },
  iconError: {
    backgroundColor: '#fdecea',
  },
  iconNeutral: {
    backgroundColor: '#eef6ff',
  },
  iconText: {
    fontSize: 28,
  },
  title: {
    fontSize: 18,
    fontWeight: '700',
    marginBottom: 6,
    textAlign: 'center',
  },
  message: {
    fontSize: 14,
    color: '#666',
    textAlign: 'center',
    marginBottom: 14,
    paddingHorizontal: 6,
  },
  button: {
    backgroundColor: '#007bff',
    paddingHorizontal: 20,
    paddingVertical: 10,
    borderRadius: 10,
    minWidth: 120,
    alignItems: 'center',
  },
  buttonText: {
    color: '#fff',
    fontWeight: '600',
  },
  note: {
    marginTop: 10,
    fontSize: 12,
    color: '#999',
    textAlign: 'center',
  },
});
