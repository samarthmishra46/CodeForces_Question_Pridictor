// src/components/RecommendationItem.tsx
import React from 'react';
import type { ProblemRecommendation } from '../types';
import './RecommendationItem.css'; // We'll create this CSS file

interface RecommendationItemProps {
    recommendation: ProblemRecommendation;
}

const RecommendationItem: React.FC<RecommendationItemProps> = ({ recommendation }) => {
    return (
        <div className="recommendation-item">
            <h3>
                <a href={recommendation.url} target="_blank" rel="noopener noreferrer">
                    {recommendation.name} ({recommendation.id || 'N/A'})
                </a>
            </h3>
            <p><strong>Rating:</strong> {recommendation.rating ?? 'N/A'}</p>
            <p><strong>Tags:</strong> {recommendation.tags.length > 0 ? recommendation.tags.join(', ') : 'None'}</p>
            <p><strong>Justification:</strong> {recommendation.justification}</p>
        </div>
    );
};

export default RecommendationItem;