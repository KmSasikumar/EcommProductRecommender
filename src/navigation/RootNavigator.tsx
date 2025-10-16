// navigation/RootNavigator.tsx
import React from 'react';
import { NavigationContainer } from '@react-navigation/native';
import { createStackNavigator } from '@react-navigation/stack';
import HomeScreen from '../screens/HomeScreen';
import SearchPage from '../screens/SearchPage';
import SearchResultsScreen from '../screens/SearchResultsScreen';

// Define the stack navigator's param list
export type RootStackParamList = {
  Home: undefined; // No parameters for Home screen
  SearchPage: undefined; // No parameters for SearchPage (middle page)
  SearchResults: { initialQuery: string }; // SearchResults screen expects an initial query for NCF
};

const Stack = createStackNavigator<RootStackParamList>();

export default function RootNavigator() {
  return (
    <NavigationContainer>
      <Stack.Navigator 
        initialRouteName="Home"
        screenOptions={{
          headerShown: false, // Hide default headers since we're using custom headers
        }}
      >
        <Stack.Screen 
          name="Home" 
          component={HomeScreen}
        />
        <Stack.Screen 
          name="SearchPage" 
          component={SearchPage}
        />
        <Stack.Screen 
          name="SearchResults" 
          component={SearchResultsScreen}
        />
      </Stack.Navigator>
    </NavigationContainer>
  );
}