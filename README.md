# SHL Assessment Recommendation System

## Overview

This project implements an intelligent recommendation system designed to help hiring managers find relevant SHL assessments based on job requirements. It addresses the limitations of simple keyword searches by leveraging a Retrieval-Augmented Generation (RAG) pipeline powered by Large Language Models (LLMs). The system accepts natural language queries, including constraints like skills, duration, job level, or even URLs pointing to job descriptions, and recommends suitable assessments from SHL's product catalog.

This project was developed as part of the SHL AI Intern RE Generative AI assignment.

**Live Demo:** [https://shl-recommender-777265733443.us-central1.run.app/](https://shl-recommender-777265733443.us-central1.run.app/)
**API Endpoint:** [https://shl-recommender-777265733443.us-central1.run.app/api/recommend](https://shl-recommender-777265733443.us-central1.run.app/api/recommend)

## Features

* **Natural Language Query Understanding:** Interprets user queries using Google's Gemini LLM (`gemini-2.0-flash-001`) to extract skills, roles, categories, duration limits, and other filters.
* **Job Description URL Processing:** Attempts to fetch and extract text from job description URLs (using `requests` and `BeautifulSoup`) to incorporate into the query. (Note: This feature is experimental and may be unreliable for sites with anti-scraping measures like LinkedIn).
* **Hybrid Retrieval:** Combines multiple strategies for finding relevant assessments:
    * **Metadata Filtering:** Applies strict filters based on extracted constraints (duration, category, job level, remote/adaptive support).
    * **Explicit Text Filtering:** Ensures core keywords (normalized and matched as whole words) appear in the assessment name or description.
    * **Semantic Search:** Uses sentence embeddings (`all-MiniLM-L6-v2`) and a FAISS index to find assessments semantically similar to the query.
* **Ranking & Boosting:** Ranks combined results, prioritizing filtered candidates and boosting those with explicit keyword matches.
* **Package Recommendation:** For multi-skill queries, identifies the best combination of assessments (up to 3) that maximizes skill coverage within the specified duration limit. Returns partial packages with notes if full coverage isn't feasible.
* **Web Interface:** A simple Flask-based web UI for submitting queries and viewing results in a table format.
* **JSON API:** A GET endpoint (`/api/recommend`) for programmatic access, returning recommendations in JSON format.
* **Containerized Deployment:** Packaged using Docker for consistent deployment (currently deployed on Google Cloud Run).

## Architecture/Approach

The system follows a Retrieval-Augmented Generation (RAG) pattern:

1.  **Data Preprocessing:** Assessment data from the SHL catalog was cleaned, structured, and saved as `cleaned_assessments_metadata.json`. Embeddings were generated and stored in a FAISS index (`cleaned_assessments.index`).
2.  **Query Input:** The Flask app receives a query via the UI or API.
3.  **URL Handling (if applicable):** URLs are detected; content is fetched/parsed (experimental). Text is combined.
4.  **Query Understanding:** The combined text is sent to the Gemini LLM to extract structured `search_keywords` and `filters`. Keywords are normalized and expanded.
5.  **Hybrid Retrieval:**
    * `filter_metadata`: Applies structured filters and explicit keyword text matching.
    * `semantic_search`: Queries the FAISS index using expanded keywords.
6.  **Ranking/Boosting:** Filtered and semantic results are merged, ranked (filter priority + semantic score), and boosted based on keyword presence.
7.  **Package Finding:** If multiple skills were requested, combinations of ranked results are evaluated for optimal skill coverage within constraints.
8.  **Output Formatting:** The final list of recommendations (either individual, full package, or partial package) is formatted for display (Web UI) or as JSON (API), including relevant metadata and contextual notes.

## Technology Stack

* **Language:** Python 3.12
* **Web Framework:** Flask
* **Core AI/ML:**
    * Google Gemini API (`gemini-2.0-flash-001`)
    * `sentence-transformers` (`all-MiniLM-L6-v2`)
    * `faiss-cpu`
    * `torch` / `transformers`
* **Data Handling:** `json`, `numpy`
* **Web Scraping:** `requests`, `beautifulsoup4`
* **Deployment:** Docker, Gunicorn, Google Cloud Run, GitHub
* **Environment:** `python-dotenv`

## Data Files

* `data/cleaned_assessments_metadata.json`: Contains the preprocessed metadata for 367 SHL assessments.
* `data/cleaned_assessments.index`: The FAISS index built from the assessment embeddings.

## Setup and Local Execution

1.  **Clone Repository:**
    ```bash
    git clone [https://github.com/raakshassh/SHL-recommender.git](https://github.com/raakshassh/SHL-recommender.git)
    cd SHL-recommender
    ```
2.  **Create Virtual Environment:**
    ```bash
    python -m venv venv
    # Activate (Windows)
    venv\Scripts\activate
    # Activate (macOS/Linux)
    source venv/bin/activate
    ```
3.  **Install PyTorch:** Install PyTorch separately first, following instructions for your OS/CUDA from [pytorch.org](https://pytorch.org/get-started/locally/).
    * *Example (CPU only):* `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu`
4.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    ```
5.  **Set Environment Variables:** Create a `.env` file in the project root and add your API key:
    ```env
    GEMINI_API_KEY=YOUR_ACTUAL_API_KEY
    ```
6.  **Run Flask App:**
    ```bash
    flask run --host=0.0.0.0 --port=5001
    ```
    Access the web app at `http://localhost:5001`.

## API Endpoint Usage

* **URL:** `/api/recommend`
* **Method:** GET
* **Query Parameter:** `query` (URL-encoded natural language query or query with JD URL)
* **Example (`curl`):**
    ```bash
    curl "[https://shl-recommender-777265733443.us-central1.run.app/api/recommend?query=Python%2C%20SQL%2C%20JavaScript%20package%20under%2060%20mins](https://www.google.com/search?q=https://shl-recommender-777265733443.us-central1.run.app/api/recommend%3Fquery%3DPython%252C%2520SQL%252C%2520JavaScript%2520package%2520under%252060%2520mins)"
    ```
* **Success Response (Example Structure):**
    ```json
    {
      "recommendations": [
        {
          "Name": "Assessment Name",
          "URL": "http://...",
          "Duration": "X minutes",
          "Remote Testing Support": "Yes", // or "No" / "N/A"
          "Adaptive/IRT Support": "Yes", // or "No" / "N/A"
          "Test Types": ["Type A", "Type B"]
        },
        // ... more recommendations ...
      ],
      "result_type": "package", // or "partial_package", "individual", "none"
      "note": "Optional note explaining the result type or missing skills."
    }
    ```
* **Error Response:** Returns JSON with an `error` key and appropriate HTTP status code (400 for bad request, 500 for internal error, 503 if service unavailable).

## Deployment

The application is containerized using Docker (`Dockerfile`) and configured to run with Gunicorn. It is currently deployed on **Google Cloud Run**. Deployment typically involves:
1.  Pushing code (including `Dockerfile`) to the GitHub repository.
2.  Connecting the repository to Google Cloud Run.
3.  Configuring the Cloud Run service to build from the `Dockerfile`.
4.  Setting the `GEMINI_API_KEY` as a secret environment variable in Cloud Run.
5.  Cloud Run builds the image and deploys the service.

