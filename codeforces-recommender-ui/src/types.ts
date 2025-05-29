// src/types.ts
export interface ProblemRecommendation {
    id: string;
    name: string;
    url: string;
    rating?: number | string | null; // Can be number, "N/A", or null
    tags: string[];
    justification: string;
}