import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AppProvider } from './contexts/AppContext';
import MainLayout from './components/layout/MainLayout';
import Dashboard from './pages/Dashboard';
import TreeExplorer from './pages/TreeExplorer';
import CounterfactualAnalysis from './pages/CounterfactualAnalysis';
import Settings from './pages/Settings';
import './styles/index.css';

function App() {
  return (
    <AppProvider>
      <Router>
        <MainLayout>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/tree/:treeId" element={<TreeExplorer />} />
            <Route path="/counterfactuals/:treeId" element={<CounterfactualAnalysis />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </MainLayout>
      </Router>
    </AppProvider>
  );
}

export default App;