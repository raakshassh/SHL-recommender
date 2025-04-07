import os
import json
import traceback # Import traceback for detailed error logging
import re # Import regex for URL detection
import requests # For fetching URL content
from bs4 import BeautifulSoup # For parsing HTML
from flask import Flask, request, render_template, jsonify, abort
from dotenv import load_dotenv
from typing import Optional # Import Optional for type hinting

# Import the refactored pipeline module
import recommendation_engine as engine # Use alias 'engine'

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# --- Initialize Pipeline Resources ---
pipeline_ready = False
try:
    engine.initialize_pipeline()
    pipeline_ready = True
except Exception as e:
    print(f"FATAL: Pipeline initialization failed during Flask startup: {e}")
    print(traceback.format_exc())

# --- Web Scraping Helper ---

def fetch_and_extract_text(url: str) -> Optional[str]: # Optional type hint used here
    """
    Fetches content from a URL and extracts text using BeautifulSoup.
    Uses basic heuristics to find main content - LIKELY NEEDS SITE-SPECIFIC TUNING.
    """
    print(f"Attempting to fetch and extract text from: {url}")
    try:
        # Add a user-agent header to mimic a browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10) # Add timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx or 5xx)

        # Check content type - only parse HTML
        content_type = response.headers.get('content-type', '').lower()
        if 'html' not in content_type:
            print(f"WARN: Content type is not HTML ({content_type}). Skipping extraction for {url}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')

        # --- Basic Text Extraction Logic ---
        # This is generic and might not work well for specific sites like LinkedIn.
        # Try common tags/attributes for main content. Needs refinement.
        main_content = soup.find('main') or \
                       soup.find('article') or \
                       soup.find('div', attrs={'role': 'main'}) or \
                       soup.find('body') # Fallback to body

        if main_content:
            # Remove script and style tags first
            for script_or_style in main_content(['script', 'style']):
                script_or_style.decompose()
            # Get text, separate paragraphs/divs with spaces, strip extra whitespace
            text = main_content.get_text(separator=' ', strip=True)
            # Limit length? Optional.
            # text = text[:5000] # Limit to first 5000 chars?
            print(f"Successfully extracted ~{len(text)} characters from {url}")
            return text
        else:
            print(f"WARN: Could not find main content container in {url}. Returning None.")
            return None

    except requests.exceptions.RequestException as e:
        print(f"ERROR: Failed to fetch URL {url}: {e}")
        return None
    except Exception as e:
        print(f"ERROR: Failed to parse or extract text from {url}: {e}")
        print(traceback.format_exc())
        return None

# --- URL Detection Helper ---
URL_REGEX = r'https?://\S+'

def detect_and_process_query(query_text: str) -> str:
    """
    Detects URLs in the query, fetches content, and combines text.
    Returns the text to be processed by the recommendation engine.
    """
    urls = re.findall(URL_REGEX, query_text)
    processed_text = query_text

    if urls:
        # For simplicity, use the first URL found
        url_to_fetch = urls[0]
        print(f"Detected URL: {url_to_fetch}")

        # Remove the URL from the original query text to get remaining constraints
        remaining_text = re.sub(URL_REGEX, '', query_text).strip()
        print(f"Remaining text query: '{remaining_text}'")

        # Fetch and extract text from the URL
        extracted_jd_text = fetch_and_extract_text(url_to_fetch)

        if extracted_jd_text:
            # Combine extracted text with remaining query parts
            processed_text = f"{remaining_text}\n\nJob Description Content:\n{extracted_jd_text}"
            print("INFO: Combined remaining query text with extracted URL content.")
        else:
            # If fetching failed, use only the remaining text
            print("WARN: Failed to extract text from URL. Using only the non-URL part of the query.")
            processed_text = remaining_text
    else:
        # No URL detected, use the original query
        processed_text = query_text

    # Return the text ready for the pipeline
    # Add a check for empty processed_text
    if not processed_text or processed_text.isspace():
        print("WARN: Processed text is empty after URL handling.")
        return "" # Return empty string to avoid errors downstream

    return processed_text


# --- Flask Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """Handles the main web page: displays form and results."""
    if not pipeline_ready:
        return render_template('index.html', error="Recommendation engine initialization failed. Please check server logs.")

    results = None
    query = ""
    error = None

    if request.method == 'POST':
        query = request.form.get('query', '').strip()
        if not query:
            error = "Please enter a query."
        else:
            try:
                # Process query (handles URL detection and fetching)
                processed_query_text = detect_and_process_query(query)

                if not processed_query_text:
                     error = "Query became empty after processing URL (if any). Please provide more details."
                     results = None
                else:
                    print(f"Processing combined/final text in pipeline: '{processed_query_text[:200]}...'")
                    results = engine.run_pipeline(processed_query_text)
                    print(f"Pipeline returned: {results}")

            except Exception as e:
                print(f"ERROR during pipeline execution for query '{query}': {e}")
                print(traceback.format_exc())
                error = "An error occurred while processing your query."
                results = None

    return render_template('index.html', results=results, query=query, error=error)


@app.route('/api/recommend', methods=['GET'])
def api_recommend():
    """Handles API requests for recommendations."""
    if not pipeline_ready:
        abort(503, description="Recommendation engine is not available.")

    query = request.args.get('query', '').strip()
    if not query:
        return jsonify({"error": "Missing 'query' parameter"}), 400

    try:
        # Process query (handles URL detection and fetching)
        processed_query_text = detect_and_process_query(query)

        if not processed_query_text:
            return jsonify({"error": "Query is empty after processing URL (if any)."}), 400

        print(f"Processing combined/final text in API pipeline: '{processed_query_text[:200]}...'")
        result_data = engine.run_pipeline(processed_query_text)
        return jsonify(result_data)

    except Exception as e:
        print(f"ERROR during API pipeline execution for query '{query}': {e}")
        print(traceback.format_exc())
        return jsonify({"error": "An internal error occurred"}), 500


# --- Run the App ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    # Set debug=True for development ONLY if needed, recommended False otherwise
    app.run(debug=False, host='0.0.0.0', port=port)
