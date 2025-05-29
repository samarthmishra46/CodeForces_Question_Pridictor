// src/components/RecommendationList.tsx
import React from 'react';
import type { ProblemRecommendation } from '../types';
import RecommendationItem from './RecommendationItem';
import './RecommendationList.css'; // We'll create this

interface RecommendationListProps {
    recommendations: ProblemRecommendation[];
}

const RecommendationList: React.FC<RecommendationListProps> = ({ recommendations }) => {
    if (recommendations.length === 0) {
        return <p className="no-recommendations">No recommendations found for the given criteria.</p>;
    }

    // Handle special case: error messages from backend formatted as recommendations
    if (recommendations.length === 1 && recommendations[0].id.startsWith("ERROR_") || recommendations[0].id === "NO_CANDIDATES" || recommendations[0].id === "FALLBACK_EMPTY") {
        return <p className="error-message">{recommendations[0].justification || recommendations[0].name}</p>
    }


    return (
        <div className="recommendation-list">
            <h2>Recommendations:</h2>
            {recommendations.map((rec, index) => (
                <RecommendationItem key={rec.id + '-' + index} recommendation={rec} />
            ))}
        </div>
    );
};

export default RecommendationList;