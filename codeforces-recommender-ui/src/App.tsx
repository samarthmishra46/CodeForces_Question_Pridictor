// src/App.tsx
import React, { useState } from 'react';
import type { FormEvent } from 'react';
import type { ProblemRecommendation } from './types';
import RecommendationList from './components/RecommendationList';
import './App.css';

// The base URL for your FastAPI backend
// You might want to use an environment variable for this in a real app
const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000';

function App() {
    const [username, setUsername] = useState<string>('');
    const [forceRefresh, setForceRefresh] = useState<boolean>(false);
    const [recommendations, setRecommendations] = useState<ProblemRecommendation[] | null>(null);
    const [isLoading, setIsLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();
        if (!username.trim()) {
            setError('Please enter a Codeforces username.');
            setRecommendations(null);
            return;
        }

        setIsLoading(true);
        setError(null);
        setRecommendations(null);

        try {
            const url = `${API_BASE_URL}/recommendations/${encodeURIComponent(username.trim())}?force_refresh=${forceRefresh}`;
            const response = await fetch(url);

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || `API Error: ${response.status}`);
            }

            const data: ProblemRecommendation[] = await response.json();
            setRecommendations(data);

        } catch (err: any) {
            setError(err.message || 'Failed to fetch recommendations.');
            setRecommendations(null); // Clear any old recommendations
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Codeforces Growth Recommender</h1>
            </header>
            <main>
                <form onSubmit={handleSubmit} className="input-form">
                    <div className="form-group">
                        <label htmlFor="username">Codeforces Username:</label>
                        <input
                            type="text"
                            id="username"
                            value={username}
                            onChange={(e) => setUsername(e.target.value)}
                            placeholder="e.g., tourist"
                            required
                        />
                    </div>
                    <div className="form-group form-group-checkbox">
                        <input
                            type="checkbox"
                            id="forceRefresh"
                            checked={forceRefresh}
                            onChange={(e) => setForceRefresh(e.target.checked)}
                        />
                        <label htmlFor="forceRefresh">Force Refresh User Status</label>
                    </div>
                    <button type="submit" disabled={isLoading}>
                        {isLoading ? 'Loading...' : 'Get Recommendations'}
                    </button>
                </form>

                {error && <p className="error-message main-error">{error}</p>}

                {recommendations && <RecommendationList recommendations={recommendations} />}
                
                {!isLoading && !recommendations && !error && (
                    <p className="info-message">Enter a username to get recommendations.</p>
                )}
            </main>
            <footer className="App-footer">
                <p>Powered by AI and Codeforces API</p>
            </footer>
        </div>
    );
}

export default App;