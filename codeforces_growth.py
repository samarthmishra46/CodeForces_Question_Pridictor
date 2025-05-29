#codeforces_growth.py

import requests
import json
import os
import random
import time
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
import logging # Added for better logging

# --- Logging Setup ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- LLM SDKs ---
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None
    logger.info("OpenAI SDK not found. OpenAI features will be disabled.")
try:
    import anthropic
except ImportError:
    anthropic = None
    logger.info("Anthropic SDK not found. Anthropic features will be disabled.")
try:
    import google.generativeai as genai
except ImportError:
    genai = None
    logger.info("Google Generative AI SDK not found. Google features will be disabled.")

# --- Configuration ---
CODEFORCES_API_BASE_URL = "https://codeforces.com/api/"

# Cache paths relative to this script's directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_BASE_DIR = os.path.join(SCRIPT_DIR, ".cache") # Store cache in a hidden subdir
PROBLEMS_CACHE_FILE_CF = os.path.join(CACHE_BASE_DIR, "codeforces_problems_cache.json")
USER_STATUS_CACHE_DIR_CF = os.path.join(CACHE_BASE_DIR, "cf_user_status_cache")

USER_STATUS_CACHE_DURATION_SECONDS = 60 * 10  # 10 minutes for user status cache
PROBLEMS_CACHE_DURATION_SECONDS = 86400 * 7 # 7 days for problems cache

class CodeforcesGrowthRecommender:
    def __init__(self, llm_preference: List[str] = None, num_recommendations_llm: int = 3, num_candidates_llm: int = 20, num_fallback_candidates: int = 3):
        """
        Initialize the recommender.
        llm_preference: Order of LLMs to try, e.g., ["google", "anthropic", "openai"]
        """
        os.makedirs(CACHE_BASE_DIR, exist_ok=True) # Ensure base cache directory exists
        os.makedirs(USER_STATUS_CACHE_DIR_CF, exist_ok=True)

        self.all_cf_problems, self.all_cf_problem_stats = self._get_all_codeforces_problems()
        self.llm_preference = llm_preference if llm_preference else ["google", "anthropic", "openai"]
        self.num_recommendations_llm = num_recommendations_llm
        self.num_candidates_llm = num_candidates_llm
        self.num_fallback_candidates = num_fallback_candidates

        logger.info("CodeforcesGrowthRecommender Initialized.")
        if not self.all_cf_problems:
            logger.warning("Could not load Codeforces problems. Recommendations might be limited or fail.")
        if not any([OpenAI, anthropic, genai]):
            logger.warning("No LLM SDKs (openai, anthropic, google-generativeai) are installed. LLM features will be disabled.")

    # --- Codeforces API Fetching ---
    def _cf_api_request(self, endpoint, params=None) -> Optional[Any]:
        url = CODEFORCES_API_BASE_URL + endpoint
        try:
            response = requests.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json()
            if data.get("status") == "OK":
                return data.get("result")
            else:
                logger.error(f"Codeforces API Error ({endpoint}): {data.get('comment', 'Unknown error')}")
                return None
        except requests.exceptions.RequestException as e:
            logger.error(f"Request failed for {endpoint}: {e}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Failed to decode JSON from {endpoint}. Response: {response.text if 'response' in locals() else 'No response object'}")
            return None

    def _get_codeforces_user_info(self, handle: str) -> Optional[List[Dict]]:
        # User info is small and can change (rating), so not caching it aggressively here,
        # but Codeforces API might have its own internal caching.
        # If this endpoint is hit very frequently for the same user, consider a short cache.
        return self._cf_api_request("user.info", params={"handles": handle})

    def _get_codeforces_user_status(self, handle: str, count: int = 3000, force_refresh: bool = False) -> Optional[List[Dict]]:
        cache_file = os.path.join(USER_STATUS_CACHE_DIR_CF, f"{handle}_status.json")

        if force_refresh:
            logger.info(f"Force refresh requested for user status: {handle}")
            if os.path.exists(cache_file):
                try:
                    os.remove(cache_file)
                    logger.info(f"Removed existing cache file for {handle} due to force refresh.")
                except OSError as e:
                    logger.error(f"Error removing cache file for {handle} during force refresh: {e}")
        
        if not force_refresh and os.path.exists(cache_file):
            try:
                if time.time() - os.path.getmtime(cache_file) < USER_STATUS_CACHE_DURATION_SECONDS:
                    with open(cache_file, 'r') as f:
                        logger.info(f"Loading cached status for {handle} (valid for {USER_STATUS_CACHE_DURATION_SECONDS // 60} mins).")
                        return json.load(f)
                else:
                    logger.info(f"User status cache for {handle} expired. Fetching fresh data.")
            except Exception as e:
                logger.error(f"Error reading cache for {handle}: {e}. Fetching fresh data.")
        
        logger.info(f"Fetching status for {handle} from Codeforces API...")
        result = self._cf_api_request("user.status", params={"handle": handle, "count": count})
        if result:
            try:
                with open(cache_file, 'w') as f:
                    json.dump(result, f)
                logger.info(f"Cached user status for {handle}.")
            except IOError as e:
                logger.error(f"Error caching user status for {handle}: {e}")
        return result

    def _get_all_codeforces_problems(self) -> Tuple[List[Dict], List[Dict]]:
        if os.path.exists(PROBLEMS_CACHE_FILE_CF):
            try:
                if time.time() - os.path.getmtime(PROBLEMS_CACHE_FILE_CF) < PROBLEMS_CACHE_DURATION_SECONDS:
                    with open(PROBLEMS_CACHE_FILE_CF, 'r') as f:
                        data = json.load(f)
                    if data and data.get("problems") and data.get("problemStatistics"):
                        logger.info(f"Loaded {len(data['problems'])} Codeforces problems from cache.")
                        return data['problems'], data['problemStatistics']
                else:
                    logger.info("Codeforces problems cache expired. Fetching fresh data.")
            except (json.JSONDecodeError, IOError) as e:
                logger.error(f"Error loading CF problems from cache: {e}. Fetching from API.")

        logger.info("Fetching all Codeforces problems from API (this might take a moment)...")
        result = self._cf_api_request("problemset.problems")
        if result and "problems" in result and "problemStatistics" in result:
            try:
                with open(PROBLEMS_CACHE_FILE_CF, 'w') as f:
                    json.dump(result, f)
                logger.info(f"Cached {len(result['problems'])} Codeforces problems to {PROBLEMS_CACHE_FILE_CF}.")
                return result["problems"], result["problemStatistics"]
            except IOError as e:
                logger.error(f"Error caching CF problems: {e}")
                # Still return the fetched data if caching failed
                return result["problems"], result["problemStatistics"]
        
        logger.warning("Could not fetch problems from Codeforces API. Using a small fallback.")
        return [
            {"contestId": 1, "index": "A", "name": "Theatre Square", "type": "PROGRAMMING", "points": 500.0, "rating": 800, "tags": ["math", "implementation"]},
            {"contestId": 71, "index": "A", "name": "Way Too Long Words", "type": "PROGRAMMING", "rating": 800, "tags": ["strings", "implementation"]},
        ], [{"contestId": 1, "index": "A", "solvedCount": 100000}, {"contestId": 71, "index": "A", "solvedCount": 200000}]

    # --- Profile Analysis & Candidate Selection ---
    def _analyze_and_select_cf_candidates(self, user_info_list: Optional[List[Dict]], user_status: Optional[List[Dict]], num_candidates: int) -> Tuple[str, List[Dict]]:
        if not user_info_list or not self.all_cf_problems:
            return "User info or problem list is empty.", []
        user_info = user_info_list[0]
        current_rating = user_info.get("rating", 0)
        max_rating = user_info.get("maxRating", 0)
        effective_rating = current_rating
        if current_rating < 1000 and max_rating > current_rating + 200: effective_rating = (current_rating + max_rating) // 2
        elif current_rating == 0: effective_rating = 800

        solved_problem_ids = set()
        solved_problem_ratings = []
        solved_tags_counter = Counter()
        if user_status:
            for sub in user_status:
                if sub.get("verdict") == "OK":
                    problem = sub["problem"]
                    problem_id = f"{problem.get('contestId', '')}{problem.get('index', '')}"
                    solved_problem_ids.add(problem_id)
                    if problem.get("rating"): solved_problem_ratings.append(problem["rating"])
                    for tag in problem.get("tags", []): solved_tags_counter[tag] += 1
        else:
            logger.warning(f"User status is None for {user_info.get('handle', 'unknown_user')}. Analysis will assume no problems solved.")
        
        # Rating window logic
        if effective_rating < 1200:
            rating_lower_bound = max(800, effective_rating - 100)
            rating_upper_bound = effective_rating + 200
        elif effective_rating < 1600:
            rating_lower_bound = effective_rating
            rating_upper_bound = effective_rating + 200
        elif effective_rating < 2000:
            rating_lower_bound = effective_rating + 0
            rating_upper_bound = effective_rating + 300
        else:
            rating_lower_bound = effective_rating + 100
            rating_upper_bound = effective_rating + 300
        if rating_upper_bound < rating_lower_bound + 100: rating_upper_bound = rating_lower_bound + 100
        if rating_lower_bound < 800: rating_lower_bound = 800
        
        profile_summary = (
            f"User: {user_info['handle']}\n"
            f"Current CF Rating: {user_info.get('rating', 'Unrated')} (Max: {max_rating}, Rank: {user_info.get('rank', 'N/A')})\n"
            f"Effective Rating for Recommendations: {effective_rating}\n"
            f"Total Distinct Solved (in fetched status): {len(solved_problem_ids)}\n"
            f"Solved Problem Ratings (sample of last 5 if available): {solved_problem_ratings[-5:]}\n"
            f"Most Practiced Tags: {solved_tags_counter.most_common(5)}\n"
            f"Target Growth Problem Rating Window: {rating_lower_bound} - {rating_upper_bound}."
        )
        logger.info(f"Profile summary for {user_info['handle']}:\n{profile_summary}")
        
        scored_candidates = []
        for prob in self.all_cf_problems:
            prob_id = f"{prob.get('contestId', '')}{prob.get('index', '')}"
            prob_rating = prob.get("rating")
            if prob_id not in solved_problem_ids and prob_rating:
                score = 0
                if rating_lower_bound - 100 <= prob_rating <= rating_upper_bound + 100: score += 1
                if rating_lower_bound <= prob_rating <= rating_upper_bound:
                    score += 5
                    if prob_rating > effective_rating and prob_rating <= rating_upper_bound: score += 3
                prob_tags = prob.get("tags", [])
                if prob_tags:
                    if any(tag not in solved_tags_counter for tag in prob_tags): score += 2
                    if any(solved_tags_counter.get(tag, 0) < 3 for tag in prob_tags): score += 2
                is_new_tag_for_user = any(tag not in solved_tags_counter for tag in prob_tags)
                if prob_rating < effective_rating - 100 and not is_new_tag_for_user: score -= 3
                
                if score > 0:
                    prob_with_url = prob.copy()
                    prob_with_url['url'] = f"https://codeforces.com/problemset/problem/{prob.get('contestId')}/{prob.get('index')}"
                    scored_candidates.append({"problem": prob_with_url, "score": score})
        
        scored_candidates.sort(key=lambda x: x["score"], reverse=True)
        candidate_problems_for_llm = [sc["problem"] for sc in scored_candidates[:num_candidates]]
        return profile_summary, candidate_problems_for_llm

    # --- LLM Integration ---
    def _get_llm_prompt(self, profile_summary: str, candidate_problems_list: List[Dict]) -> str:
        candidate_details = "\n".join([
            (f"- Name: \"{p['name']}\" (ID: {p.get('contestId', '')}{p.get('index', '')}), "
             f"Rating: {p.get('rating', 'N/A')}, Tags: {', '.join(p.get('tags', ['N/A']))}, "
             f"URL: {p.get('url', 'N/A')}")
            for p in candidate_problems_list
        ])
        # Prompt remains the same as before, ensuring JSON output request
        return f"""
        You are an expert Codeforces coach. Your task is to recommend {self.num_recommendations_llm} practice problems
        for a user based on their Codeforces profile and a list of candidate problems.
        The goal is to suggest problems that will help the user GROW. This means problems that are:
        1. CHALLENGING BUT SOLVABLE: Within their target growth problem rating window, ideally pushing them slightly.
        2. EDUCATIONAL: Covering topics/tags they need to improve, explore, or solidify at a new difficulty.
        3. PROGRESSIVE: Helping them move towards their next rating milestone.

        Do NOT recommend problems that are likely too easy (significantly below their effective rating unless it's for a brand new tag)
        or overwhelmingly hard (far above their target growth window). The user has already solved problems listed in their profile,
        so you should only choose from the UNSOLVED candidate problems provided below.

        User Profile Summary (Codeforces):
        {profile_summary}

        Candidate Problems (These are UNSOLVED problems. Choose from this list. Each problem includes its URL. Consider their 'Rating' as difficulty and the user's target window):
        {candidate_details}

        Please select exactly {self.num_recommendations_llm} problems from the list above.
        For each problem, provide:
        - Problem Name (ID)
        - Rating
        - Tags
        - URL (use the provided URL for the chosen problem)
        - Justification: Explain clearly (1-2 sentences) HOW this specific problem will help the user grow, considering their profile.

        Format your response strictly as a JSON list of objects. Each object must have these keys: "problem_name_id", "rating", "tags", "url", "justification".
        Example:
        [
          {{
            "problem_name_id": "Problem Name 1 (ContestID1Index1)",
            "rating": 1600,
            "tags": ["dp", "graphs"],
            "url": "https://codeforces.com/problemset/problem/ContestID1/Index1",
            "justification": "This problem focuses on dynamic programming on graphs, which seems to be an area you can strengthen based on your current rating and solved tags."
          }}
        ]
        """

    def _parse_llm_json_response(self, llm_response_text: str) -> List[Dict]:
        try:
            json_start = llm_response_text.find('[')
            json_end = llm_response_text.rfind(']')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = llm_response_text[json_start : json_end+1]
            else:
                json_str = llm_response_text

            parsed_data = json.loads(json_str)
            if not isinstance(parsed_data, list):
                logger.warning("LLM response is not a list after JSON parsing.")
                return []

            recommendations = []
            for item in parsed_data:
                if isinstance(item, dict) and \
                   all(k in item for k in ["problem_name_id", "rating", "tags", "url", "justification"]):
                    name_id_str = item["problem_name_id"]
                    problem_id = ""
                    problem_name = name_id_str
                    if '(' in name_id_str and name_id_str.endswith(')'):
                        problem_name = name_id_str[:name_id_str.rfind('(')].strip()
                        problem_id = name_id_str[name_id_str.rfind('(')+1:-1]

                    recommendations.append({
                        "id": problem_id,
                        "name": problem_name,
                        "rating": item["rating"],
                        "tags": item["tags"] if isinstance(item["tags"], list) else [item["tags"]],
                        "url": item["url"],
                        "justification": item["justification"]
                    })
                else:
                    logger.warning(f"Skipping malformed item in LLM JSON response: {item}")
            return recommendations
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}. Response was: \n{llm_response_text[:500]}...")
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing LLM JSON: {e}")
            return []

    def _call_openai(self, full_prompt: str, api_key: str) -> str:
        if not OpenAI: return "OpenAI library not installed."
        try:
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert Codeforces coach. Respond in JSON format as specified."},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=1000, temperature=0.3, response_format={ "type": "json_object" }
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return f"Error calling OpenAI API: {e}"

    def _call_anthropic(self, full_prompt: str, api_key: str) -> str:
        if not anthropic: return "Anthropic library not installed."
        try:
            client = anthropic.Anthropic(api_key=api_key)
            system_prompt = "You are an expert Codeforces coach. Your output must be strictly a JSON list of objects as per the user's instructions. Do not include any other text before or after the JSON."
            message = client.messages.create(
                model="claude-3-haiku-20240307",
                max_tokens=1000, system=system_prompt,
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.3
            )
            return message.content[0].text.strip()
        except Exception as e:
            logger.error(f"Error calling Anthropic API: {e}")
            return f"Error calling Anthropic API: {e}"

    def _call_google(self, full_prompt: str, api_key: str) -> str:
        if not genai: return "Google Generative AI library not installed."
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash-latest",
                system_instruction="You are an expert Codeforces coach. Your output must be strictly a JSON list of objects as per the user's instructions. Do not include any other text before or after the JSON."
            )
            response = model.generate_content(
                full_prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=1000, temperature=0.3, response_mime_type="application/json"
                )
            )
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error calling Google API: {e}")
            return f"Error calling Google API: {e}"

    def get_recommendations(self, cf_username: str, force_refresh_status: bool = False) -> List[Dict]:
        logger.info(f"Processing request for Codeforces user: {cf_username}{' (forcing status refresh)' if force_refresh_status else ''}")

        user_info_data = self._get_codeforces_user_info(cf_username)
        if not user_info_data:
            logger.error(f"Could not fetch user info for {cf_username}.")
            return [{"id": "ERROR_USER_NOT_FOUND", "name": "User Not Found or API Error", "url": "", "justification": f"Could not fetch user info for {cf_username}."}]

        user_status_data = self._get_codeforces_user_status(cf_username, force_refresh=force_refresh_status)
        # Note: user_status_data can be None if API fails, _analyze_and_select_cf_candidates should handle it.
        
        if not self.all_cf_problems:
             logger.error("Core problem data could not be loaded. Cannot generate recommendations.")
             return [{"id": "ERROR_PROBLEM_DATA", "name": "Problem Data Unavailable", "url": "", "justification": "Core problem data could not be loaded."}]

        profile_summary, candidate_problems_for_llm = self._analyze_and_select_cf_candidates(
            user_info_data, user_status_data, self.num_candidates_llm
        )

        if isinstance(profile_summary, str) and not candidate_problems_for_llm: # Error message returned
            logger.error(f"Error during analysis for {cf_username}: {profile_summary}")
            return [{"id": "ERROR_ANALYSIS", "name": "Analysis Error", "url": "", "justification": profile_summary}]
        
        if not candidate_problems_for_llm:
            logger.warning(f"No suitable candidate problems found for {cf_username} after filtering.")
            return [{"id": "NO_CANDIDATES", "name": "No Candidate Problems", "url": "", "justification": "Could not find suitable unsolved problems for your profile in the target range."}]

        logger.info(f"Selected {len(candidate_problems_for_llm)} candidate problems for LLM input for {cf_username}.")
        
        llm_response_text = "LLM not attempted or failed."
        prompt = self._get_llm_prompt(profile_summary, candidate_problems_for_llm)
        llm_used = None

        for provider_name in self.llm_preference:
            api_key_env_var = f"{provider_name.upper()}_API_KEY"
            api_key = os.environ.get(api_key_env_var)
            if not api_key:
                logger.info(f"{api_key_env_var} not set. Skipping {provider_name}.")
                continue

            logger.info(f"Attempting LLM provider: {provider_name}...")
            if provider_name == "google" and genai:
                llm_response_text = self._call_google(prompt, api_key)
                llm_used = "Google Gemini"
            elif provider_name == "anthropic" and anthropic:
                llm_response_text = self._call_anthropic(prompt, api_key)
                llm_used = "Anthropic Claude"
            elif provider_name == "openai" and OpenAI:
                llm_response_text = self._call_openai(prompt, api_key)
                llm_used = "OpenAI GPT"
            else:
                logger.warning(f"Provider {provider_name} selected but SDK not available or check failed.")
                continue
            
            is_error_response = "error calling" in llm_response_text.lower() or \
                                "not installed" in llm_response_text.lower() or \
                                "api key" in llm_response_text.lower() # Check for generic error messages

            if not is_error_response and llm_response_text: # Added check for empty response
                logger.info(f"Successfully received response from {llm_used}.")
                break 
            else:
                logger.warning(f"Failed with {llm_used or provider_name}: {llm_response_text[:200]}...") # Log snippet of error
                llm_used = None
        
        if llm_used:
            logger.info(f"Raw LLM Response from {llm_used} (first 500 chars):\n{llm_response_text[:500]}...")
            parsed_recommendations = self._parse_llm_json_response(llm_response_text)
            if parsed_recommendations:
                logger.info(f"Successfully parsed {len(parsed_recommendations)} recommendations from LLM for {cf_username}.")
                return parsed_recommendations
            else:
                logger.warning(f"Failed to parse LLM response for {cf_username}. Falling back.")
        else:
            logger.warning(f"All configured LLM attempts failed or no LLMs available for {cf_username}. Falling back.")
        
        logger.info(f"Falling back to rule-based candidate selection for {cf_username}.")
        fallback_recs = []
        for p_data in candidate_problems_for_llm[:self.num_fallback_candidates]:
            fallback_recs.append({
                "id": f"{p_data.get('contestId', '')}{p_data.get('index', '')}",
                "name": p_data.get("name", "Unknown Problem"),
                "url": p_data.get("url", ""),
                "rating": p_data.get("rating"),
                "tags": p_data.get("tags", []),
                "justification": "This is a good candidate based on your profile and rating. (LLM processing failed or was unavailable)"
            })
        
        if fallback_recs:
            logger.info(f"Generated {len(fallback_recs)} fallback recommendations for {cf_username}.")
            return fallback_recs
        else:
             logger.warning(f"LLM failed and no fallback candidates found for {cf_username}.")
             return [{"id": "FALLBACK_EMPTY", "name": "No Fallback Problems", "url": "", "justification": "LLM failed and no fallback candidates found."}]


# --- Main function for standalone script testing (Optional) ---
if __name__ == "__main__":
    logger.info("Codeforces Problem Recommender (Standalone Test Mode)")
    logger.info("Ensure API keys are set as environment variables (e.g., GOOGLE_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)")

    recommender = CodeforcesGrowthRecommender(llm_preference=["google", "anthropic", "openai"])

    test_handle = input("Enter Codeforces handle for testing (e.g., tourist): ").strip()
    if test_handle:
        force_refresh_input = input("Force refresh user status from API? (yes/no) [no]: ").strip().lower()
        force_refresh = force_refresh_input == 'yes'

        recommendations = recommender.get_recommendations(test_handle, force_refresh_status=force_refresh)
        
        print("\n--- Recommendations ---") # Using print here for clear test output
        if recommendations:
            for rec in recommendations:
                print(f"- Name: {rec.get('name')} (ID: {rec.get('id')})")
                print(f"  URL: {rec.get('url')}")
                print(f"  Rating: {rec.get('rating')}")
                print(f"  Tags: {rec.get('tags')}")
                print(f"  Justification: {rec.get('justification')}")
                print("-" * 20)
        else:
            print("No recommendations generated.")
    else:
        logger.info("No handle provided for testing.")