// src/services/firebase.ts
import { initializeApp } from 'firebase/app';
import { getDatabase, push, ref } from 'firebase/database';
import { getFunctions } from 'firebase/functions';

const firebaseConfig = {
  apiKey: "AIzaSyAbSCIKcQnGGZoI9EmAI4co8kNg7-pd5y8",
  authDomain: "ecomm-recommender-rn.firebaseapp.com",
  projectId: "ecomm-recommender-rn",
  storageBucket: "ecomm-recommender-rn.firebasestorage.app",
  messagingSenderId: "595815053031",
  appId: "1:595815053031:web:e764c35fce512d71c5d442",
};

const app = initializeApp(firebaseConfig);
export const rtdb = getDatabase(app);
export const functions = getFunctions(app, 'us-central1');
