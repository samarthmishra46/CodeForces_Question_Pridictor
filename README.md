# Codeforces Growth Recommender

Welcome to the Codeforces Growth Recommender! This project aims to help competitive programming students, particularly those on Codeforces, accelerate their learning curve by providing personalized problem recommendations. Instead of randomly picking problems or getting stuck on problems that are too easy or too hard, this tool analyzes user performance and suggests problems that are optimally challenging and relevant to their growth areas.

**Live Demo:**
*   **Frontend:** [https://code-forces-question-pridictor-bpuy-ten.vercel.app/](https://code-forces-question-pridictor-bpuy-ten.vercel.app/)
*   **(Backend API is used by the frontend)**

## How This Project Can Leverage Your Coding Journey

Competitive programming is a fantastic way to sharpen problem-solving skills, master data structures and algorithms, and prepare for technical interviews. However, navigating the vast sea of problems on platforms like Codeforces can be daunting. This recommender helps by:

1.  **Personalized Learning Path:** Get problem suggestions tailored to your current Codeforces rating, recently solved problems, and areas where you might need improvement.
2.  **Optimal Challenge:** Recommends problems that are slightly above your comfort zone ("Zone of Proximal Development") to encourage growth without causing excessive frustration.
3.  **Targeted Skill Improvement:** By analyzing your submission history (implicitly, through problem selection criteria), the system can guide you towards diversifying your skills across different problem types and tags.
4.  **Motivation and Consistency:** Having a clear set of problems to tackle can boost motivation and help maintain a consistent practice schedule.
5.  **Efficient Practice:** Spend less time searching for what to solve and more time actually solving and learning.
6.  **Understanding Weaknesses (Implicitly):** While not explicitly a weakness analyzer, the types of problems recommended can give you hints about areas you might not have explored as much.

This tool acts as a smart study buddy, guiding you towards problems that will most effectively help you improve your Codeforces rating and overall algorithmic thinking.

## Features

*   **Personalized Recommendations:** Enter your Codeforces username to get tailored problem suggestions.
*   **LLM-Powered Analysis (Backend):** Utilizes Large Language Models to analyze problem data and user profiles for more nuanced recommendations (this is a core idea of the backend).
*   **Rating-Based Suggestions:** Considers your current Codeforces rating to suggest appropriate difficulty levels.
*   **Problem Details:** Shows problem name, URL, rating, and tags.
*   **Justification:** Provides a brief reason why a particular problem is recommended.
*   **Force Refresh Option:** Allows fetching the latest user status from Codeforces, bypassing any cache.

## Tech Stack

*   **Frontend:** React, TypeScript, Vite
*   **Backend:** Python, FastAPI
*   **LLM Integration:** (Google Gemini, Anthropic Claude, OpenAI GPT - configurable)
*   **Deployment:**
    *   Frontend: Vercel
    *   Backend: Render

## Setting Up and Running Locally

To run this project on your local system, you'll need to set up both the frontend and the backend.

### Prerequisites

*   **Node.js and npm/yarn/pnpm:** For the frontend (LTS version recommended).
*   **Python:** For the backend (Python 3.8+ recommended).
*   **Git:** For cloning the repository.
*   **Code Editor:** VS Code, PyCharm, etc.
*   **(Optional but Recommended) Virtual Environment Tool for Python:** `venv` or `conda`.
*   **API Keys for LLMs:** You'll need API keys from Google (Gemini), Anthropic (Claude), and/or OpenAI if you want to use the LLM-powered recommendation features.

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY_NAME.git
cd YOUR_REPOSITORY_NAME
```
### 2. Backend Setup
Navigate to the backend directory (if you have separate frontend and backend folders in your repository, otherwise these steps are from the project root if main.py is there).
```bash
# cd backend # If you have a dedicated backend folder

# Create and activate a Python virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt

# Create a .env file in the backend directory
# Add your API keys and other configurations:
touch .env
```
Your backend/.env file should look something like this:
```bash
# LLM API Keys (get these from the respective providers)
GOOGLE_API_KEY="your_google_api_key_here"
ANTHROPIC_API_KEY="your_anthropic_api_key_here"
OPENAI_API_KEY="your_openai_api_key_here"

# LLM Preference (comma-separated, order matters)
LLM_PREFERENCE="google,anthropic,openai" # Or just one, e.g., "google"

# Recommender Configuration (optional, defaults are in the code)
# NUM_RECOMMENDATIONS_LLM=3
# NUM_CANDIDATES_LLM=20
# NUM_FALLBACK_CANDIDATES=5

# For FastAPI development server (Uvicorn)
# HOST=0.0.0.0 # Not usually needed for local, 127.0.0.1 is default
# PORT=8000   # Default port for the backend
```
### Run the Backend Server:
From the backend directory (where main.py for FastAPI is located):
```bash
uvicorn main:app --reload --port 8000
```
The backend API should now be running, typically at http://127.0.0.1:8000.
###3. Frontend Setup
Navigate to the frontend directory (e.g., frontend/).
```bash
# cd frontend # If you have a dedicated frontend folder

# Install frontend dependencies
npm install
# or
# yarn install
# or
# pnpm install
```
This will typically start the frontend application, and it should open in your browser (e.g., at http://localhost:5173 for Vite, or http://localhost:3000 for Create React App).
### 4. Using the Application Locally
*   **Ensure both the backend and frontend servers are running.**
*   **Open the frontend URL (e.g., http://localhost:5173) in your browser.
*   **Enter a Codeforces username and get recommendations!
### Contributing
Contributions are welcome! If you have ideas for improvements, new features, or bug fixes, please feel free to:
Fork the repository.
Create a new branch (git checkout -b feature/YourAmazingFeature).
Make your changes.
Commit your changes (git commit -m 'Add some AmazingFeature').
Push to the branch (git push origin feature/YourAmazingFeature).
Open a Pull Request.
Please ensure your code follows the project's coding style and include tests if applicable.
Future Scope / Ideas
More sophisticated user profile analysis (e.g., identifying specific weak topics).
Tracking solved recommended problems.
Allowing users to specify problem tags they want to focus on.
Visualizations of progress or skill distribution.
Integration with other competitive programming platforms.

### License
***This project is licensed under the MIT License - see the LICENSE.md file for details (you'll need to create this file if you want a specific license).***
Happy Coding and Happy Growing on Codeforces!
**Remember to:**

1.  **Replace `YOUR_USERNAME/YOUR_REPOSITORY_NAME.git`** with your actual GitHub repository URL.
2.  **Create a `LICENSE.md` file** if you want to include a license (MIT is a common permissive one).
3.  **Verify directory structure:** The instructions for `cd backend` or `cd frontend` assume you might have these as subdirectories. If your `main.py` (FastAPI) and `package.json` (React) are in the root of the repository, adjust the `cd` commands or remove them.
4.  **Ensure `requirements.txt`** is accurate for your backend.
5.  **Ensure `package.json`** lists all frontend dependencies.

This README provides a good starting point. You can expand on any section as you see fit!
