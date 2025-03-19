import React, { createContext, useContext, useState, useReducer } from 'react';

const AppContext = createContext();

const initialState = {
  trees: {},
  currentTree: null,
  counterfactuals: {},
  impacts: {},
  settings: {
    temperature: 0.7,
    maxTokens: 1000,
    maxDepth: 3,
    minProbability: 0.1,
  },
  loading: false,
  error: null
};

function appReducer(state, action) {
  switch (action.type) {
    case 'SET_LOADING':
      return { ...state, loading: action.payload, error: null };
    case 'SET_ERROR':
      return { ...state, error: action.payload, loading: false };
    case 'ADD_TREE':
      return {
        ...state,
        trees: { ...state.trees, [action.payload.id]: action.payload },
        currentTree: action.payload.id,
        loading: false,
        error: null
      };
    case 'SET_CURRENT_TREE':
      return { ...state, currentTree: action.payload };
    case 'ADD_COUNTERFACTUALS':
      return {
        ...state,
        counterfactuals: {
          ...state.counterfactuals,
          [action.payload.treeId]: action.payload.counterfactuals
        },
        loading: false
      };
    case 'ADD_IMPACTS':
      return {
        ...state,
        impacts: {
          ...state.impacts,
          [action.payload.treeId]: action.payload.impacts
        },
        loading: false
      };
    case 'UPDATE_SETTINGS':
      return {
        ...state,
        settings: { ...state.settings, ...action.payload }
      };
    default:
      return state;
  }
}

export function AppProvider({ children }) {
  const [state, dispatch] = useReducer(appReducer, initialState);

  return (
    <AppContext.Provider value={{ state, dispatch }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppContext() {
  return useContext(AppContext);
}