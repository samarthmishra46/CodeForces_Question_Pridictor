import os
from fastapi import FastAPI, Query, HTTPException
from fastapi.concurrency import run_in_threadpool
from typing import List, Optional, Any, Union
from pydantic import BaseModel
import logging

# Import CORS middleware
from fastapi.middleware.cors import CORSMiddleware # <<<<<<< ADD THIS IMPORT

# Import your recommender class and its logger
from codeforces_growth import CodeforcesGrowthRecommender, logger as recommender_logger

# --- Pydantic Models for API Request/Response ---
class ProblemRecommendation(BaseModel):
    id: str
    name: str
    url: str
    rating: Optional[Any] = None
    tags: List[str] = []
    justification: str

# --- FastAPI App Setup ---
app = FastAPI(
    title="Codeforces Growth Recommender API",
    description="Provides personalized Codeforces problem recommendations to help users grow.",
    version="1.0.0"
)

# --- CORS Middleware Setup ---
# This should be placed before you define your routes/endpoints.
# Define the list of origins that are allowed to make cross-origin requests.
# For development, you might use a wildcard "*" or specific local origins.
# For production, restrict this to your frontend's actual domain(s).
origins = [
    "http://localhost",         # If your frontend is served from http://localhost (no port)
    "http://localhost:3000",    # Common for React dev server
    "http://localhost:5173",    # Common for Vite/Vue dev server
    "http://localhost:8080",    # Common for Vue CLI dev server
    "http://127.0.0.1:5500",
    "https://code-forces-question-pridictor-bpuy-ten.vercel.app/",
            # Common for VS Code Live Server
    # Add THE SPECIFIC ORIGIN (PROTOCOL + HOSTNAME + PORT) of your frontend
    # If you are unsure, for testing, you can use ["*"] but this is insecure for production.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # Allows specific origins
    # allow_origins=["*"], # Allows all origins - USE WITH CAUTION FOR TESTING ONLY
    allow_credentials=True, # Allows cookies to be included in cross-origin requests
    allow_methods=["*"],    # Allows all HTTP methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],    # Allows all headers
)
# --- END CORS Middleware Setup ---


# --- Initialize Recommender ---
llm_preference_str = os.getenv("LLM_PREFERENCE", "google,anthropic,openai")
llm_preference_list = [p.strip() for p in llm_preference_str.split(',') if p.strip()]

def get_int_env(var_name: str, default: int) -> int:
    try:
        return int(os.getenv(var_name, str(default)))
    except ValueError:
        recommender_logger.warning(
            f"Invalid value for environment variable {var_name}. "
            f"Expected an integer, got '{os.getenv(var_name)}'. Using default: {default}."
        )
        return default

NUM_RECOMMENDATIONS_LLM = get_int_env("NUM_RECOMMENDATIONS_LLM", 3)
NUM_CANDIDATES_LLM = get_int_env("NUM_CANDIDATES_LLM", 20)
NUM_FALLBACK_CANDIDATES = get_int_env("NUM_FALLBACK_CANDIDATES", 5)

recommender: Optional[CodeforcesGrowthRecommender] = None
try:
    recommender = CodeforcesGrowthRecommender(
        llm_preference=llm_preference_list,
        num_recommendations_llm=NUM_RECOMMENDATIONS_LLM,
        num_candidates_llm=NUM_CANDIDATES_LLM,
        num_fallback_candidates=NUM_FALLBACK_CANDIDATES
    )
    recommender_logger.info("FastAPI app: CodeforcesGrowthRecommender initialized successfully.")
except Exception as e:
    recommender_logger.error(f"FastAPI app: Failed to initialize CodeforcesGrowthRecommender: {e}", exc_info=True)

# --- API Endpoints ---

@app.on_event("startup")
async def startup_event():
    if recommender is None:
        recommender_logger.critical(
            "FastAPI app startup: Recommender could not be initialized. "
            "API will return errors for recommendation requests."
        )
    else:
        recommender_logger.info("FastAPI app startup: Recommender available.")

@app.get(
    "/recommendations/{cf_username}",
    response_model=List[ProblemRecommendation],
    summary="Get Codeforces Problem Recommendations",
    tags=["Recommendations"],
    responses={
        200: {"description": "Successful recommendations. The list might be empty or contain status messages if no specific problems are recommended."},
        400: {"description": "Invalid input (e.g., empty username)."},
        404: {"description": "User not found or core data missing for the user."},
        500: {"description": "Internal server error, recommender service issue, or LLM/API issues."}
    }
)
async def get_recommendations_endpoint(
    cf_username: str,
    force_refresh: bool = Query(False, description="Force refresh user status from Codeforces API (bypasses cache for user status). Problem data cache is still respected.")
):
    if not cf_username.strip():
        raise HTTPException(status_code=400, detail="Codeforces username cannot be empty or whitespace.")
    
    if recommender is None:
        recommender_logger.error("API: Request for user %s - Recommender not available due to initialization failure.", cf_username)
        raise HTTPException(status_code=500, detail="Recommendation service is currently unavailable. Please try again later.")

    recommender_logger.info(f"API: Request for user: {cf_username}, force_refresh: {force_refresh}")

    try:
        recommendations_data = await run_in_threadpool(
            recommender.get_recommendations,
            cf_username=cf_username,
            force_refresh_status=force_refresh
        )

        if recommendations_data:
            first_item = recommendations_data[0]
            item_id = first_item.get("id")

            if item_id == "ERROR_USER_NOT_FOUND":
                recommender_logger.warning(f"API: User {cf_username} not found or CF API error. Detail: {first_item.get('justification')}")
                raise HTTPException(status_code=404, detail=first_item.get("justification", "User not found or Codeforces API error."))
            if item_id == "ERROR_PROBLEM_DATA":
                recommender_logger.error(f"API: Core problem data unavailable for user {cf_username}. Detail: {first_item.get('justification')}")
                raise HTTPException(status_code=500, detail=first_item.get("justification", "Core problem data unavailable, cannot generate recommendations."))
            if item_id == "ERROR_ANALYSIS":
                recommender_logger.error(f"API: Error during user profile analysis for {cf_username}. Detail: {first_item.get('justification')}")
                raise HTTPException(status_code=500, detail=first_item.get("justification", "Error during user profile analysis."))
        
        # Log before returning, especially useful for debugging response structure
        recommender_logger.info(f"API: Successfully processed request for {cf_username}. Preparing to return {len(recommendations_data) if recommendations_data else 0} items.")
        recommender_logger.debug(f"API: Data being returned for {cf_username}: {recommendations_data}") # More verbose

        return recommendations_data

    except HTTPException:
        raise
    except Exception as e:
        recommender_logger.error(f"API: Unhandled exception for user {cf_username}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while processing your request. Please try again later.")

@app.get("/", include_in_schema=False)
async def root():
    return {"message": "Welcome to the Codeforces Growth Recommender API. See /docs for API usage."}

# --- To run this application (from your terminal) ---
# ... (instructions remain the same)