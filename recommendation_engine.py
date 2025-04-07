import faiss
import numpy as np
import json
import os
import time
import re # Import regex module
import traceback
from sentence_transformers import SentenceTransformer
import google.generativeai as genai
from dotenv import load_dotenv
from typing import List, Dict, Tuple, Optional, Set, Any, Union
from collections import defaultdict # For keyword expansion
from itertools import combinations # For package finding

# --- Configuration ---
# Load environment variables when the module is imported
load_dotenv()

# File paths (relative to the project root)
# Assuming data files are in a 'data' subdirectory
# Get the directory where this script resides
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_BASE_DIR, "data") # Construct path relative to script
METADATA_FILE: str = os.path.join(DATA_DIR, "cleaned_assessments_metadata.json")
FAISS_INDEX_FILE: str = os.path.join(DATA_DIR, "cleaned_assessments.index")

# Model names
# Ensure this matches the model used to CREATE the embeddings/index
EMBEDDING_MODEL_NAME: str = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME: str = 'gemini-2.0-flash-001' # User specified model
GEMINI_API_KEY: Optional[str] = os.getenv('GEMINI_API_KEY')

# Retrieval Parameters
SEMANTIC_K: int = 50 # Number of candidates from semantic search
MAX_PACKAGE_SIZE: int = 3 # Max number of tests to combine for a package
MAX_CANDIDATES_FOR_COMBINATIONS: int = 50 # Limit pool size for package finding performance

# --- Global Variables / Caching (Module Level) ---
# These will be loaded once when the module is imported by Flask
_embedding_model: Optional[SentenceTransformer] = None
_metadata_list: Optional[List[Dict[str, Any]]] = None # Holds the CLEANED metadata
_faiss_index: Optional[faiss.Index] = None
_genai_configured: bool = False
_dynamic_categories: Optional[Set[str]] = None
# V2.4: Skill Synonym Map
SKILL_SYNONYM_MAP = {
    "javascript": ["javascript", "js", "java script", "ecmascript"],
    "python": ["python", "python programming", "py"],
    "sql": ["sql", "sql server", "database", "relational database", "query language"],
    # Add more synonyms as needed
}
# Precompile regex patterns for skills for efficiency
SKILL_REGEX_PATTERNS = {}

# --- Initialization Function ---
# This function will be called once by the Flask app at startup
def initialize_pipeline():
    """Loads all necessary models and data."""
    global _embedding_model, _metadata_list, _faiss_index, _genai_configured, _dynamic_categories
    # Prevent re-initialization
    if _embedding_model and _metadata_list and _faiss_index and _genai_configured:
        print("INFO: Pipeline already initialized.")
        return

    print("--- Initializing Recommendation Pipeline ---")
    start_time = time.time()
    initialization_successful = True

    # Configure GenAI first
    _genai_configured = _configure_genai_once("initialize_pipeline")
    if not _genai_configured:
        print("CRITICAL ERROR: GenAI could not be configured. Check API Key.")
        initialization_successful = False # Mark as failed

    # Load resources
    if initialization_successful:
        _metadata_list = _load_cleaned_metadata_internal() # Use internal loader
        if _metadata_list is None: initialization_successful = False

    if initialization_successful:
        _faiss_index = _load_faiss_index_internal() # Use internal loader
        if _faiss_index is None: initialization_successful = False

    if initialization_successful:
        _embedding_model = _load_embedding_model_internal() # Use internal loader
        if _embedding_model is None: initialization_successful = False

    if initialization_successful:
        _dynamic_categories = _get_dynamic_categories_internal() # Build categories after metadata load
        # No need to mark failure if categories fail, but log it

    if not initialization_successful:
         print("CRITICAL ERROR: Failed to load one or more essential resources (metadata, index, embedding model).")
         raise RuntimeError("Pipeline initialization failed: Missing essential resources.")


    # Check consistency one last time after all loads
    if _metadata_list and _faiss_index and len(_metadata_list) != _faiss_index.ntotal:
         print(f"CRITICAL WARNING: Final check - Metadata count ({len(_metadata_list)}) != FAISS count ({_faiss_index.ntotal}). Results may be incorrect.")

    print(f"--- Pipeline Initialization Complete ({time.time() - start_time:.2f}s) ---")

# --- Helper Functions (Remain Mostly Unchanged) ---

def _configure_genai_once(function_name: str) -> bool:
    """Configures the Google Generative AI client once."""
    global _genai_configured # Need to modify global state
    if _genai_configured: return True # Already configured
    if not GEMINI_API_KEY:
        print(f"ERROR ({function_name}): GEMINI_API_KEY missing in environment variables.")
        return False
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        print(f"DEBUG ({function_name}): GenAI configured.")
        _genai_configured = True # Set state here
        return True
    except Exception as e:
        print(f"ERROR ({function_name}): Configure Gemini failed: {e}")
        _genai_configured = False # Ensure state is false on error
        return False

def get_query_embedding(query_text: str, model: SentenceTransformer) -> Optional[np.ndarray]:
    """Generates a normalized embedding vector for the query."""
    if not query_text or not query_text.strip():
        print("WARN: Empty query text provided for embedding.")
        return None
    if model is None:
        print("ERROR: Embedding model not loaded.")
        return None
    try:
        embedding = model.encode([query_text.strip()], normalize_embeddings=True)
        embedding_float32 = embedding.astype(np.float32)
        if embedding_float32.ndim == 1:
            embedding_float32 = embedding_float32.reshape(1, -1)
        if embedding_float32.ndim != 2 or embedding_float32.shape[0] != 1:
            raise ValueError(f"Unexpected embedding shape: {embedding_float32.shape}")
        return embedding_float32
    except Exception as e:
        print(f"ERROR: Failed generate embedding for '{query_text[:50]}...': {e}")
        return None

def search_faiss(index: faiss.Index, query_embedding: np.ndarray, k: int, metadata_count: int) -> Tuple[List[float], List[int]]:
    """Performs a search on the FAISS index."""
    if index is None:
        print("ERROR: FAISS index not loaded.")
        return [], []
    if query_embedding is None or query_embedding.ndim != 2 or query_embedding.shape[0] != 1:
        print(f"ERROR: Invalid query embedding shape: {query_embedding.shape if query_embedding is not None else 'None'}")
        return [], []
    if query_embedding.shape[1] != index.d:
        print(f"ERROR: Query embedding dimension ({query_embedding.shape[1]}) != Index dimension ({index.d}).")
        return [], []
    if k <= 0:
        print("WARN: Search k must be > 0.")
        return [], []
    try:
        k_actual = min(k, index.ntotal)
        if k_actual == 0:
            print("WARN: Index is empty, cannot search.")
            return [], []
        distances, indices = index.search(query_embedding, k_actual)
        result_indices = indices[0]
        result_scores = distances[0]
        valid_mask = (result_indices != -1) & (result_indices < metadata_count)
        valid_indices = result_indices[valid_mask].tolist()
        valid_scores = result_scores[valid_mask].tolist()
        if len(valid_indices) != len(indices[0]):
            print(f"WARN: Some FAISS indices were invalid or out of bounds.")
        return valid_scores, valid_indices
    except Exception as e:
        print(f"ERROR: FAISS search failed: {e}")
        print(traceback.format_exc())
        return [], []

def _compile_skill_regex(skills: List[str]):
    """Precompiles regex patterns for skills for efficiency."""
    global SKILL_REGEX_PATTERNS
    SKILL_REGEX_PATTERNS = {} # Reset
    if not skills: return # No skills to compile
    print(f"DEBUG: Compiling regex for skills: {skills}")
    for skill in skills:
        if not skill: continue # Skip empty strings
        try:
            # Use word boundary for accurate matching
            SKILL_REGEX_PATTERNS[skill] = re.compile(r'\b' + re.escape(skill) + r'\b', re.IGNORECASE)
        except re.error as e:
            print(f"WARN: Invalid regex pattern for skill '{skill}'. Cannot use for matching. Error: {e}")


# --- Internal Loading Functions (Called by initialize_pipeline) ---
# Added _internal suffix to avoid potential name clashes if this module is imported elsewhere

def _load_cleaned_metadata_internal(filepath: str = METADATA_FILE) -> Optional[List[Dict[str, Any]]]:
    """Loads the cleaned metadata list (internal)."""
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: Cleaned metadata file not found: '{filepath}'")
        return None
    print(f"INFO: Loading cleaned metadata from '{filepath}'...")
    start_time = time.time()
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError("Metadata file content is not a JSON list.")
        count = len(data)
        if count == 0: print("WARN: No records loaded from metadata file.")
        print(f"INFO: Cleaned metadata loaded ({count} items). Time: {time.time() - start_time:.2f}s")
        return data
    except Exception as e:
        print(f"ERROR: Failed to load or process metadata from '{filepath}': {e}")
        print(traceback.format_exc())
        return None

def _load_faiss_index_internal(filepath: str = FAISS_INDEX_FILE) -> Optional[faiss.Index]:
    """Loads the FAISS index (internal)."""
    if not os.path.exists(filepath):
        print(f"CRITICAL ERROR: FAISS index file not found: '{filepath}'.")
        return None
    print(f"INFO: Loading FAISS index from '{filepath}'...")
    start_time = time.time()
    try:
        index = faiss.read_index(filepath)
        print(f"INFO: FAISS index loaded ({index.ntotal} vectors, dim={index.d}). Time: {time.time() - start_time:.2f}s")
        return index
    except Exception as e:
        print(f"ERROR: Failed to load FAISS index '{filepath}': {e}")
        return None

def _load_embedding_model_internal() -> Optional[SentenceTransformer]:
    """Loads the Sentence Transformer model (internal)."""
    print(f"INFO: Loading embedding model: '{EMBEDDING_MODEL_NAME}'...")
    start_time = time.time()
    try:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        print(f"INFO: Embedding model loaded. Time: {time.time() - start_time:.2f}s")
        return model
    except Exception as e:
        print(f"ERROR: Failed load embedding model '{EMBEDDING_MODEL_NAME}': {e}")
        # If model loading fails, raise error to stop initialization
        raise e # Re-raise the exception

def _get_dynamic_categories_internal() -> Set[str]:
    """Builds category set from the cleaned metadata (internal)."""
    if _metadata_list is None:
        print("WARN: Cannot build categories, metadata not loaded yet.")
        return set()
    print("INFO: Building dynamic category set from loaded metadata...")
    categories: Set[str] = set()
    try:
        for item in _metadata_list:
            cats = item.get('test_type_names_list', [])
            if isinstance(cats, list):
                categories.update(c.lower() for c in cats if isinstance(c, str) and c.strip())
        print(f"INFO: Dynamic categories built ({len(categories)} unique).")
        return categories
    except Exception as e:
        print(f"ERROR: Failed build dynamic categories: {e}")
        return set()

# --- V2.4 Skill Normalization ---
def normalize_and_map_skills(keywords: List[str]) -> List[str]:
    """Normalizes extracted keywords using a synonym map."""
    normalized_skills = set()
    processed_keywords = set()
    synonym_to_canonical = {}
    for canonical, synonyms in SKILL_SYNONYM_MAP.items():
        for syn in synonyms:
            synonym_to_canonical[syn.lower()] = canonical

    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in processed_keywords: continue
        canonical_skill = synonym_to_canonical.get(kw_lower)
        if canonical_skill:
            normalized_skills.add(canonical_skill)
            processed_keywords.update(s.lower() for s in SKILL_SYNONYM_MAP.get(canonical_skill, [kw_lower]))
        else:
            normalized_skills.add(kw_lower)
            processed_keywords.add(kw_lower)

    final_skills = sorted(list(normalized_skills))
    if final_skills != keywords:
         print(f"DEBUG: Normalized skills from {keywords} to {final_skills}")
    _compile_skill_regex(final_skills) # Compile regex after normalization
    return final_skills


# --- V2 Keyword Expansion ---
def expand_keywords(keywords: List[str]) -> List[str]:
    """Expands core keywords with related terms."""
    expansions = defaultdict(lambda: [], {
        "java": ["java programming", "core java", "java development", "jvm"],
        "python": ["python programming", "scripting", "pandas", "numpy"],
        "sql": ["database", "query language", "relational database"],
        "javascript": ["frontend", "web development", "js", "ecmascript"],
        "analyst": ["analysis", "data analysis", "reporting"],
        "customer service": ["support", "client interaction", "communication"],
        "collaboration": ["teamwork", "communication"],
    })
    expanded_set = set(keywords)
    for kw in keywords:
        expanded_set.update(expansions[kw])
    print(f"DEBUG: Expanded keywords from {keywords} to {list(expanded_set)}")
    return sorted(list(expanded_set))


# --- Core Pipeline Functions (Use loaded global resources) ---

def query_understanding(user_query: str) -> Optional[Dict[str, Any]]:
    """
    Uses LLM (Gemini Flash) to understand the user query and extract structured entities.
    V2.4: Includes skill normalization after extraction.
    """
    print(f"\n--- Query Understanding ---")
    if not _genai_configured:
        print("ERROR: GenAI not configured. Cannot run query understanding.")
        return None
    if _dynamic_categories is None:
        print("WARN: Dynamic categories not loaded. Prompt may be less effective.")
        category_sample = "N/A"
    else:
        category_sample = ", ".join(sorted(list(_dynamic_categories))[:25])

    default_structure = {
        "search_keywords": [],
        "filters": { "categories": [], "job_levels": [], "duration_max_minutes": None, "remote_testing_required": None, "adaptive_testing_required": None }
    }
    default_json_string = json.dumps(default_structure)
    prompt = f"""Analyze the following user query seeking SHL assessments. Extract relevant information and structure it EXACTLY as a JSON object matching the format below.

User Query: "{user_query}"

Reference Assessment Categories (use these for 'filters.categories', match case-insensitively and output lowercase): {category_sample}
Reference Job Levels (examples): Director, Entry-Level, Executive, General Population, Graduate, Manager, Mid-Professional, Front Line Manager, Supervisor, Professional Individual Contributor

Required JSON Output Format:
```json
{default_json_string}
```

Instructions:
1. Identify CORE keywords related to skills (e.g., "Java", "SQL", "Python", "Java Script", "collaboration"), job roles ("analyst"), context, or assessment topics ("verbal reasoning"). Populate `search_keywords` with these core terms (lowercase). Be precise with skill names like "Java Script".
2. Identify specific filtering requirements:
    - Map requested assessment types (like "cognitive test", "personality test") to the 'Reference Assessment Categories'. Populate `filters.categories` (list of lowercase strings).
    - Extract requested job levels. Populate `filters.job_levels` (list of strings, preserve case if possible).
    - If a maximum duration is mentioned (e.g., "within 45 mins", "under 1 hour", "30 minutes max"), extract the maximum duration IN MINUTES as an integer. Populate `filters.duration_max_minutes` (integer or null).
    - Determine if remote testing is explicitly required (true/false/null). Populate `filters.remote_testing_required`.
    - Determine if adaptive/IRT testing is explicitly required (true/false/null). Populate `filters.adaptive_testing_required`.
3. **Output ONLY the JSON object enclosed in ```json ... ```.** No other text, explanations, or formatting.

Extracted JSON:
"""
    try:
        print(f"INFO: Calling Gemini ({LLM_MODEL_NAME}) for query understanding...")
        model = genai.GenerativeModel(LLM_MODEL_NAME)
        safety_settings = [ {"category": c, "threshold": "BLOCK_MEDIUM_AND_ABOVE"} for c in ["HARM_CATEGORY_HARASSMENT", "HARM_CATEGORY_HATE_SPEECH", "HARM_CATEGORY_SEXUALLY_EXPLICIT", "HARM_CATEGORY_DANGEROUS_CONTENT"]]
        response = model.generate_content(prompt, safety_settings=safety_settings)

        if not response.parts:
            feedback = response.prompt_feedback
            reason = feedback.block_reason if feedback else 'Unknown'
            safety_ratings_info = f" Safety Ratings: {feedback.safety_ratings}" if feedback and feedback.safety_ratings else ""
            print(f"WARN: Query understanding LLM call blocked or failed. Reason: {reason}.{safety_ratings_info}")
            return None

        llm_text = response.text
        print(f"DEBUG: Raw query understanding response:\n{llm_text}")

        json_match = re.search(r"```json\s*(\{.*?\})\s*```", llm_text, re.DOTALL | re.IGNORECASE)
        json_string = None
        if json_match:
            json_string = json_match.group(1).strip()
            print("DEBUG: Found ```json ... ``` block.")
        else:
            start_index = llm_text.find('{')
            end_index = llm_text.rfind('}')
            if start_index != -1 and end_index != -1 and end_index > start_index:
                json_string = llm_text[start_index : end_index + 1].strip()
                print("DEBUG: Found {...} block (fallback).")
            else:
                try:
                    parsed_entities_risky = json.loads(llm_text)
                    if isinstance(parsed_entities_risky, dict):
                         print("DEBUG: Parsed JSON directly from response (no markers found).")
                         json_string = llm_text
                    else: return None
                except json.JSONDecodeError:
                     print("WARN: Could not find or parse JSON block in LLM response.")
                     return None

        try:
            parsed_entities = json.loads(json_string)
            if not isinstance(parsed_entities, dict):
                print(f"WARN: Parsed result is not a dictionary: {type(parsed_entities)}")
                return None

            validated_entities = {"search_keywords": [], "filters": {}}
            raw_keywords = parsed_entities.get('search_keywords', [])
            extracted_keywords = sorted(list(set(kw.strip().lower() for kw in raw_keywords if isinstance(kw, str) and kw.strip()))) if isinstance(raw_keywords, list) else []
            validated_entities['search_keywords'] = normalize_and_map_skills(extracted_keywords)
            _compile_skill_regex(validated_entities['search_keywords']) # Compile regex for normalized skills
            core_normalized_keywords = validated_entities['search_keywords']
            validated_entities['expanded_search_keywords'] = expand_keywords(core_normalized_keywords)

            raw_filters = parsed_entities.get('filters', {})
            validated_filters = {}
            raw_cats = raw_filters.get('categories', [])
            if isinstance(raw_cats, list) and _dynamic_categories is not None:
                 valid_cats = {cat.strip().lower() for cat in raw_cats if isinstance(cat, str) and cat.strip().lower() in _dynamic_categories}
                 validated_filters['categories'] = sorted(list(valid_cats))
            else: validated_filters['categories'] = []

            raw_levels = raw_filters.get('job_levels', [])
            validated_filters['job_levels'] = sorted(list(set( lvl.strip() for lvl in raw_levels if isinstance(lvl, str) and lvl.strip() ))) if isinstance(raw_levels, list) else []

            raw_duration = raw_filters.get('duration_max_minutes')
            validated_filters['duration_max_minutes'] = None
            if isinstance(raw_duration, int) and raw_duration > 0: validated_filters['duration_max_minutes'] = raw_duration
            elif isinstance(raw_duration, str):
                 duration_match = re.search(r'\d+', raw_duration)
                 if duration_match:
                      try: validated_filters['duration_max_minutes'] = int(duration_match.group(0))
                      except ValueError: pass # Keep as None

            raw_remote = raw_filters.get('remote_testing_required')
            validated_filters['remote_testing_required'] = raw_remote if isinstance(raw_remote, bool) else None
            raw_adaptive = raw_filters.get('adaptive_testing_required')
            validated_filters['adaptive_testing_required'] = raw_adaptive if isinstance(raw_adaptive, bool) else None
            validated_entities['filters'] = validated_filters

            print(f"INFO: Query understanding successful:")
            print(json.dumps(validated_entities, indent=2))
            return validated_entities

        except json.JSONDecodeError as e:
            print(f"WARN: Failed to parse JSON from LLM response: {e}")
            print(f"DEBUG: Attempted to parse: {json_string}")
            return None

    except Exception as e:
        print(f"ERROR: Exception during query understanding: {e}")
        print(traceback.format_exc())
        return None


def semantic_search(entities: Dict, k: int) -> Dict[int, float]:
    """
    Performs semantic search using FAISS based on *expanded* keywords.
    Uses globally loaded resources.
    """
    print("\n--- Semantic Search ---")
    if not _faiss_index or not _embedding_model or _metadata_list is None:
         print("ERROR: Resources not loaded for semantic search.")
         return {}

    metadata_count = len(_metadata_list)
    semantic_scores: Dict[int, float] = {}
    search_terms = entities.get('expanded_search_keywords', entities.get('search_keywords', []))

    if not search_terms:
        print("INFO: No keywords provided for semantic search.")
        return semantic_scores

    combined_query = " ".join(search_terms)
    print(f"INFO: Performing semantic search for combined terms: '{combined_query}'")

    query_embedding = get_query_embedding(combined_query, _embedding_model)
    if query_embedding is None:
        print("ERROR: Failed to generate query embedding for semantic search.")
        return semantic_scores

    scores, indices = search_faiss(_faiss_index, query_embedding, k, metadata_count)

    for i, idx in enumerate(indices):
        semantic_scores[idx] = scores[i]

    print(f"INFO: Semantic search retrieved scores for {len(semantic_scores)} candidates.")
    return semantic_scores

def filter_metadata(entities: Dict) -> Set[int]:
    """
    Filters metadata based on structured criteria AND explicit keyword matches.
    Uses globally loaded metadata and precompiled regex patterns.
    """
    print("\n--- Metadata Filtering & Explicit Text Search ---")
    if _metadata_list is None:
        print("ERROR: Metadata not loaded, cannot filter.")
        return set()

    metadata = _metadata_list
    metadata_len = len(metadata)
    filters = entities.get('filters', {})
    core_keywords = entities.get('search_keywords', [])

    print(f"INFO: Applying metadata filters: {filters}")
    print(f"INFO: Applying explicit text filter for keywords: {core_keywords}")

    metadata_filtered_indices: Set[int] = set(range(metadata_len))

    # Step 1: Apply Metadata Filters
    if filters:
        max_duration = filters.get('duration_max_minutes')
        if max_duration is not None:
            metadata_filtered_indices = { idx for idx in metadata_filtered_indices if 0 <= idx < metadata_len and (metadata[idx].get('duration_minutes') is None or metadata[idx].get('duration_minutes') <= max_duration) }
            print(f"DEBUG: After duration filter ({max_duration} min): {len(metadata_filtered_indices)} candidates remain.")
        req_remote = filters.get('remote_testing_required')
        if req_remote is not None:
            metadata_filtered_indices = { idx for idx in metadata_filtered_indices if 0 <= idx < metadata_len and metadata[idx].get('remote_testing') == req_remote }
            print(f"DEBUG: After remote testing filter ({req_remote}): {len(metadata_filtered_indices)} candidates remain.")
        req_adaptive = filters.get('adaptive_testing_required')
        if req_adaptive is not None:
            metadata_filtered_indices = { idx for idx in metadata_filtered_indices if 0 <= idx < metadata_len and metadata[idx].get('adaptive_testing') == req_adaptive }
            print(f"DEBUG: After adaptive testing filter ({req_adaptive}): {len(metadata_filtered_indices)} candidates remain.")
        req_categories = set(cat.lower() for cat in filters.get('categories', []))
        if req_categories:
            metadata_filtered_indices = { idx for idx in metadata_filtered_indices if 0 <= idx < metadata_len and req_categories.issubset(set(cat.lower() for cat in metadata[idx].get('test_type_names_list', []))) }
            print(f"DEBUG: After category filter ({req_categories}): {len(metadata_filtered_indices)} candidates remain.")
        req_levels = set(filters.get('job_levels', []))
        if req_levels:
            metadata_filtered_indices = { idx for idx in metadata_filtered_indices if 0 <= idx < metadata_len and req_levels.intersection(set(metadata[idx].get('job_levels', []))) }
            print(f"DEBUG: After job level filter ({req_levels}): {len(metadata_filtered_indices)} candidates remain.")
    else: print("INFO: No metadata filters applied.")

    # Step 2: Apply Explicit Text Filter (using precompiled regex)
    if not core_keywords or not SKILL_REGEX_PATTERNS:
        print("INFO: No core keywords or patterns for explicit text filtering. Skipping.")
        final_filtered_indices = metadata_filtered_indices
    else:
        text_filtered_indices = set()
        keyword_patterns = [SKILL_REGEX_PATTERNS.get(kw) for kw in core_keywords if SKILL_REGEX_PATTERNS.get(kw)]
        if not keyword_patterns:
             print("WARN: No valid regex patterns found for explicit text filtering keywords. Skipping text filter.")
             final_filtered_indices = metadata_filtered_indices
        else:
            for idx in metadata_filtered_indices:
                if not (0 <= idx < metadata_len): continue
                item_name = metadata[idx].get('name', '')
                item_desc = metadata[idx].get('description', '')
                if any( (item_name and p.search(item_name)) or (item_desc and p.search(item_desc)) for p in keyword_patterns ):
                     text_filtered_indices.add(idx)
            final_filtered_indices = text_filtered_indices
            print(f"DEBUG: After explicit text filter ({core_keywords}): {len(final_filtered_indices)} candidates remain.")

    print(f"INFO: Filtering complete. {len(final_filtered_indices)} candidates remain after all filters.")
    return final_filtered_indices


def combine_rank_boost(
    filtered_indices: Set[int],
    semantic_scores: Dict[int, float],
    entities: Dict
    ) -> List[int]:
    """
    Combines filtered results with semantic search results, ranks, and boosts.
    Uses globally loaded metadata and precompiled regex patterns.
    """
    print("\n--- Combining, Ranking, and Boosting Results ---")
    if _metadata_list is None:
        print("ERROR: Metadata not loaded for ranking.")
        return []
    metadata = _metadata_list
    metadata_len = len(metadata)
    combined_candidates = filtered_indices.union(semantic_scores.keys())
    print(f"INFO: Combined candidates (from filter & semantic): {len(combined_candidates)}")

    # Step 1: Initial Ranking
    initial_ranked_list = []
    filter_pass_candidates = []
    semantic_only_candidates = []
    for idx in combined_candidates:
        if not (0 <= idx < metadata_len): continue
        score = semantic_scores.get(idx, float('inf'))
        if idx in filtered_indices: filter_pass_candidates.append((idx, score))
        else: semantic_only_candidates.append((idx, score))
    filter_pass_candidates.sort(key=lambda x: x[1])
    semantic_only_candidates.sort(key=lambda x: x[1])
    initial_ranked_list = [idx for idx, score in filter_pass_candidates] + [idx for idx, score in semantic_only_candidates]
    print(f"INFO: Initial ranking complete. Candidate count: {len(initial_ranked_list)}")

    # Step 2: Post-Ranking Boosting
    core_keywords = entities.get('search_keywords', [])
    if not core_keywords or not SKILL_REGEX_PATTERNS:
        print("INFO: No core keywords or valid patterns for boosting. Using initial ranking.")
        final_ranked_list = initial_ranked_list
    else:
        print(f"INFO: Applying boost for whole word matches of keywords: {core_keywords}")
        boosted_indices = []
        other_indices = []
        keyword_patterns = [SKILL_REGEX_PATTERNS.get(kw) for kw in core_keywords if SKILL_REGEX_PATTERNS.get(kw)]
        if not keyword_patterns:
             print("WARN: No valid regex patterns found for boosting keywords. Skipping boost.")
             final_ranked_list = initial_ranked_list
        else:
            for idx in initial_ranked_list:
                if not (0 <= idx < metadata_len): continue
                item_name = metadata[idx].get('name', '')
                item_desc = metadata[idx].get('description', '')
                found_match = False
                for pattern in keyword_patterns:
                    if (item_name and pattern.search(item_name)) or (item_desc and pattern.search(item_desc)):
                        found_match = True; break
                if found_match: boosted_indices.append(idx)
                else: other_indices.append(idx)
            final_ranked_list = boosted_indices + other_indices
            print(f"INFO: Boosting applied. Boosted: {len(boosted_indices)}, Others: {len(other_indices)}")

    print(f"INFO: Final ranking complete. Final candidate count: {len(final_ranked_list)}")
    return final_ranked_list


# --- V2.4 Package Finding Logic (Score-Based) ---

def _get_covered_skills(combo_indices: Tuple[int], skills_required: List[str]) -> Set[str]:
    """Helper to find which required skills are covered by a combination."""
    covered_skills = set()
    if not skills_required or not SKILL_REGEX_PATTERNS or _metadata_list is None: return covered_skills
    metadata = _metadata_list
    metadata_len = len(metadata)
    combined_text = ""
    for idx in combo_indices:
        if 0 <= idx < metadata_len: combined_text += f" {metadata[idx].get('name', '')} {metadata[idx].get('description', '')}"
        else: print(f"WARN: Invalid index {idx} in skill coverage check.")
    combined_text = combined_text.lower()
    for skill in skills_required:
        pattern = SKILL_REGEX_PATTERNS.get(skill)
        if pattern and pattern.search(combined_text): covered_skills.add(skill)
    return covered_skills

def _get_total_duration(combo_indices: Tuple[int]) -> Optional[int]:
    """Helper to calculate the total duration of a test combination."""
    if _metadata_list is None: return None
    metadata = _metadata_list
    metadata_len = len(metadata)
    total_duration = 0
    for idx in combo_indices:
        if 0 <= idx < metadata_len:
            duration = metadata[idx].get('duration_minutes')
            if duration is None: return None # Cannot determine total if one is unknown
            total_duration += duration
        else:
             print(f"WARN: Invalid index {idx} in duration calculation.")
             return None
    return total_duration

def find_assessment_package(
    ranked_indices: List[int],
    entities: Dict,
    max_tests: int = MAX_PACKAGE_SIZE
    ) -> Tuple[Optional[List[int]], Set[str]]:
    """
    Attempts to find the best combination maximizing skill coverage.
    Uses globally loaded metadata and precompiled regex patterns.
    """
    print("\n--- Attempting to Find Best Assessment Package (Max Coverage) ---")
    if _metadata_list is None: return None, set()
    metadata = _metadata_list
    required_skills = entities.get('search_keywords', [])
    max_duration = entities.get('filters', {}).get('duration_max_minutes')

    if not required_skills: return None, set()
    print(f"INFO: Looking for package covering skills: {required_skills}")
    if max_duration is not None: print(f"INFO: Maximum total duration: {max_duration} minutes")

    candidate_pool_indices = ranked_indices[:MAX_CANDIDATES_FOR_COMBINATIONS]
    if not candidate_pool_indices: return None, set()

    best_combo_indices: Optional[Tuple[int]] = None
    best_covered_skills: Set[str] = set()
    best_combo_duration: Optional[int] = float('inf')
    best_combo_size: int = max_tests + 1

    for r in range(1, max_tests + 1):
        print(f"DEBUG: Checking combinations of size {r}...")
        possible_combos = combinations(candidate_pool_indices, r)
        for combo_indices_tuple in possible_combos:
            combo_duration = _get_total_duration(combo_indices_tuple)
            if max_duration is not None and (combo_duration is None or combo_duration > max_duration): continue

            covered_skills = _get_covered_skills(combo_indices_tuple, required_skills)
            num_skills_covered = len(covered_skills)

            # Skip if this combo covers 0 required skills
            if num_skills_covered == 0: continue

            is_better = False
            if num_skills_covered > len(best_covered_skills): is_better = True
            elif num_skills_covered == len(best_covered_skills): # Tie in skills covered
                 if r < best_combo_size: is_better = True # Prefer smaller package
                 elif r == best_combo_size: # Same skill count, same size
                      current_best_duration = best_combo_duration if best_combo_duration is not None else float('inf')
                      this_combo_duration = combo_duration if combo_duration is not None else float('inf')
                      if this_combo_duration < current_best_duration: is_better = True # Prefer shorter duration

            if is_better:
                print(f"DEBUG: Found new best combo (Size {r}, Duration {combo_duration}, Skills {num_skills_covered}/{len(required_skills)}): {combo_indices_tuple} -> {covered_skills}")
                best_combo_indices = combo_indices_tuple
                best_covered_skills = covered_skills
                best_combo_duration = combo_duration
                best_combo_size = r

    if best_combo_indices:
        print(f"INFO: Best package found (Size {best_combo_size}, Duration {best_combo_duration}, Skills {len(best_covered_skills)}/{len(required_skills)}): {best_combo_indices}")
        final_combo_list = sorted(list(best_combo_indices), key=lambda i: ranked_indices.index(i))
        return final_combo_list, best_covered_skills
    else:
        print("INFO: No suitable package combination found meeting constraints.")
        return None, set()


def format_output(final_indices: List[int], max_recommendations: int = 10, result_type: str = "individual", note: Optional[str] = None) -> Dict[str, Any]:
    """
    Formats the final recommendations into the required structure.
    Uses globally loaded metadata. V2.4.3: Removed default notes.
    """
    print("\n--- Formatting Final Output ---")
    if _metadata_list is None: return {"recommendations": [], "result_type": "error", "note": "Error: Metadata unavailable."}
    metadata_list = _metadata_list
    metadata_count = len(metadata_list)
    output_recommendations = []
    is_grouping = result_type in ["package", "partial_package"]
    num_to_format = len(final_indices) if is_grouping else min(len(final_indices), max_recommendations)

    type_description = "as a package" if result_type == "package" else ("as the best partial package found" if result_type == "partial_package" else "")
    print(f"INFO: Formatting {num_to_format} recommendations {type_description}...")

    final_note = note # Only use note passed from run_pipeline

    for idx in final_indices[:num_to_format]:
        if 0 <= idx < metadata_count:
            item = metadata_list[idx]
            duration_val = item.get('duration_minutes')
            duration_display_str = f"{duration_val} minutes" if isinstance(duration_val, int) else item.get('duration_display', 'N/A')
            remote_support_str = "Yes" if item.get("remote_testing") is True else "No" if item.get("remote_testing") is False else "N/A"
            adaptive_support_str = "Yes" if item.get("adaptive_testing") is True else "No" if item.get("adaptive_testing") is False else "N/A"

            output_item = {
                "Name": item.get("name", "N/A"),
                "URL": item.get("url", "#"),
                # "Description": item.get("description", ""), # Exclude description for now
                "Duration": duration_display_str,
                "Remote Testing Support": remote_support_str,
                "Adaptive/IRT Support": adaptive_support_str,
                "Test Types": item.get("test_type_names_list", []),
                # "Job Levels": item.get("job_levels", []) # Exclude job levels for now
            }
            output_recommendations.append(output_item)
        else: print(f"WARN: Invalid index {idx} during final formatting.")

    print(f"INFO: Formatted {len(output_recommendations)} recommendations.")
    return {"recommendations": output_recommendations, "result_type": result_type, "note": final_note}


# --- Main Pipeline Function (Exported for Flask) ---

def run_pipeline(user_query: str) -> Dict[str, Any]:
    """
    Runs the full RAG pipeline (V2.4.3) and returns results dictionary.
    Assumes initialize_pipeline() has been called.
    """
    pipeline_start_time = time.time()
    print(f"\n=== Running Pipeline for Query: '{user_query}' ===")
    final_note = None
    result_type = "individual"

    if not all([_metadata_list, _faiss_index, _embedding_model]):
        print("CRITICAL ERROR: Pipeline resources not initialized.")
        return {"recommendations": [], "result_type": "error", "note": "Pipeline not initialized."}

    # 1. Query Understanding
    entities = query_understanding(user_query)
    if entities is None: return {"recommendations": [], "result_type": "error", "note": "Failed to understand query."}

    # 2. Filtering
    filtered_indices = filter_metadata(entities)

    # 3. Semantic Search
    semantic_scores = semantic_search(entities, SEMANTIC_K)

    # 4. Rank & Boost
    ranked_indices = combine_rank_boost(filtered_indices, semantic_scores, entities)

    # 5. Package Finding Logic
    final_indices_for_output = ranked_indices # Default
    required_skills = entities.get('search_keywords', [])

    if len(required_skills) > 1:
        best_package_indices, covered_skills = find_assessment_package(ranked_indices, entities)
        if best_package_indices:
            final_indices_for_output = best_package_indices
            if len(covered_skills) == len(required_skills): result_type = "package"
            else:
                result_type = "partial_package"
                missing_skills = set(required_skills) - covered_skills
                # This note will be passed to format_output
                final_note = f"Found the best combination within constraints, covering: {', '.join(sorted(list(covered_skills)))}. Could not cover: {', '.join(sorted(list(missing_skills)))}."
                print(f"INFO: Partial package found. Missing skills: {missing_skills}")
        else:
            result_type = "individual"
             # This note will be passed to format_output
            final_note = "Could not find a suitable combination of tests covering the required skills within constraints. Showing top individual results."
            print("INFO: No package found. Falling back to individual ranking.")

    if not final_indices_for_output:
         print("INFO: No candidates found matching any criteria.")
          # This note will be passed to format_output
         final_note = "No relevant assessments found based on the criteria."
         result_type = "none"

    # 6. Format Output
    final_output = format_output(
        final_indices_for_output,
        result_type=result_type,
        note=final_note # Pass the generated note
    )

    print(f"\n--- Pipeline Finished ---")
    total_time = time.time() - pipeline_start_time
    print(f"Total Processing Time: {total_time:.2f}s")
    return final_output

# --- Example Execution (If run directly) ---
# This block is useful for testing the engine standalone
if __name__ == "__main__":
    print("--- Running recommendation_engine.py Standalone for Testing (V2.4.3) ---")
    # Initialize pipeline explicitly when run as script
    initialize_pipeline()

    test_queries = [
         "I want a personality and cognitive test within 45 minutes for an analyst role",
         "Need tests for Java developers, max 40 minutes",
         "Assessments for entry-level customer service roles focusing on collaboration",
         "Find tests for Python and SQL knowledge under 1 hour",
         "Show me adaptive tests for managers",
         "What tests measure verbal reasoning ability?",
         "I need a javascript test", # Test distinction from Java
         "I am hiring for Java developers who can also collaborate effectively with my business teams. Looking for an assessment(s) that can be completed in 40 minutes.", # User query 1
         "Looking to hire mid-level professionals who are proficient in Python, SQL and Java Script. Need an assessment package that can test all skills with max duration of 60 minutes." # User query 2
    ]

    # Example: Run the last test query
    if _genai_configured: # Only run if API key is valid
         test_query = test_queries[-1]
         print(f"\nRunning test query: \"{test_query}\"")
         result_data = run_pipeline(test_query)

         print("\n" + "="*25 + " Test Result " + "="*25)
         if result_data is None:
             print("Pipeline execution failed.")
         else:
             recommendations = result_data.get("recommendations", [])
             result_type = result_data.get("result_type", "error")
             note = result_data.get("note")

             if note: print(f"Note: {note}\n")
             if not recommendations:
                 if result_type != "error": print("No suitable recommendations found.")
             else:
                 if result_type == "package": print("Recommended Package (Covers All Skills):")
                 elif result_type == "partial_package": print("Recommended Partial Package (Best Coverage Found):")
                 else: print("Top Recommendations:")
                 print(json.dumps(recommendations, indent=2))
                 print(f"\n--- {len(recommendations)} recommendation(s) returned ---")
         print("="*70)
    else:
         print("\nSkipping test run because GenAI is not configured (check API key).")

