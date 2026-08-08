import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './styles/tokens.css';

const container = document.getElementById('root');
if (!container) {
  throw new Error('#root element not found');
}

createRoot(container).render(
  <StrictMode>
    <div className="j-ambient">
      <div className="j-grid" />
      <div className="j-bloom" />
      <div className="j-scanlines" />
      <div className="j-vignette" />
    </div>
    <App />
  </StrictMode>,
);
