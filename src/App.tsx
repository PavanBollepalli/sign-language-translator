import React from 'react';
import { Route, Routes } from 'react-router-dom';
import TranslatePage from './pages/TranslatePage';
import LandingPage from './pages/LandingPage';

function App() {
  return (
    <Routes>
      <Route path="/" element={<LandingPage />} />
      <Route path="/translate" element={<TranslatePage />} />
    </Routes>
  );
}

export default App;