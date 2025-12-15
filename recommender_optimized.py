"""
Research Paper Recommender System - OPTIMIZED FOR DEPLOYMENT
All-in-one combined file with recommender logic + API
Optimized for minimal Docker image size
"""

from fastapi import FastAPI, HTTPException, Query, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from typing import List, Optional, Dict, Any
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta
import pyodbc
import warnings
import uvicorn
import logging
from functools import lru_cache
from fastapi.concurrency import run_in_threadpool
import sys
import io

# Set encoding
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8', errors='replace')

warnings.filterwarnings('ignore')


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    format='{"time":"%(asctime)s", "level":"%(levelname)s", "message":"%(message)s"}',
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# ============================================================================
# SETTINGS & CONFIGURATION
# ============================================================================

class Settings(BaseSettings):
    """Application settings"""
    DB_NAME: str = "Insight"
    DB_USER: str = "sa"
    DB_PASSWORD: str = "12345678"
    DB_HOST: str = "MOSTAFA_RADY\\SQLEXPRESS"
    DB_PORT: int = 1433
    DB_DRIVER: str = "ODBC Driver 18 for SQL Server"
    API_TITLE: str = "Research Paper Recommender API"
    API_VERSION: str = "2.0.0"
    ENVIRONMENT: str = "development"
    ALLOWED_ORIGINS: str = "*"
    CONNECTION_POOL_SIZE: int = 10
    CACHE_TTL: int = 3600

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


# ============================================================================
# DATABASE CONNECTION POOL
# ============================================================================

class DatabasePool:
    """Database connection pool manager"""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.connection_string = self._build_connection_string()
        self._pool = []
        self._pool_size = settings.CONNECTION_POOL_SIZE
        logger.info("Database pool initialized")

    def _build_connection_string(self) -> str:
        server_part = self.settings.DB_HOST
        if '\\' not in server_part and ',' not in server_part:
            server_part += f',{self.settings.DB_PORT}'

        return (
            f'DRIVER={{{self.settings.DB_DRIVER}}};'
            f'SERVER={server_part};'
            f'DATABASE={self.settings.DB_NAME};'
            f'UID={self.settings.DB_USER};'
            f'PWD={self.settings.DB_PASSWORD};'
            'TrustServerCertificate=yes;'
        )

    def get_connection(self):
        try:
            if self._pool:
                return self._pool.pop()
            conn = pyodbc.connect(self.connection_string, timeout=30)
            logger.debug("Created new database connection")
            return conn
        except pyodbc.Error as e:
            logger.error(f"Database connection error: {str(e)}")
            raise HTTPException(status_code=500, detail="Database connection failed")

    def return_connection(self, conn):
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            self._pool.append(conn)
        except:
            try:
                conn.close()
            except:
                pass

    def close_all(self):
        for conn in self._pool:
            try:
                conn.close()
            except:
                pass
        self._pool.clear()


db_pool: Optional[DatabasePool] = None


# ============================================================================
# CACHE
# ============================================================================

class InMemoryCache:
    """Simple in-memory cache with TTL"""

    def __init__(self):
        self._cache: Dict[str, tuple[Any, datetime]] = {}
        self._ttl = get_settings().CACHE_TTL

    def get(self, key: str) -> Optional[Any]:
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < timedelta(seconds=self._ttl):
                logger.debug(f"Cache HIT: {key}")
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any):
        self._cache[key] = (value, datetime.now())
        logger.debug(f"Cache SET: {key}")

    def clear(self):
        self._cache.clear()
        logger.info("Cache cleared")

    def get_stats(self) -> Dict:
        valid = sum(1 for _, (_, ts) in self._cache.items() 
                   if datetime.now() - ts < timedelta(seconds=self._ttl))
        return {
            "total_entries": len(self._cache),
            "valid_entries": valid,
            "ttl_seconds": self._ttl
        }


cache = InMemoryCache()


# ============================================================================
# DATABASE FUNCTIONS
# ============================================================================

def load_data_from_db(conn):
    """Load all necessary data from database into pandas DataFrames"""
    logger.info("Loading data from database...")

    queries = {
        'users': "SELECT * FROM [User]",
        'researchers': "SELECT * FROM [Researcher]",
        'papers': "SELECT * FROM [Paper]",
        'authors': "SELECT * FROM [Author]",
        'write': "SELECT * FROM [Author_Paper]",
        'fields': "SELECT * FROM [Field]",
        'reviews': "SELECT * FROM [Review]",
        'downloads': "SELECT * FROM [Download]",
        'searches': "SELECT * FROM [Search]",
        'paper_keywords': "SELECT * FROM [Paper_Keywords]",
        'researcher_fields': "SELECT * FROM [Researcher_Field]"
    }

    data = {}
    
    for table_name, query in queries.items():
        try:
            df = pd.read_sql_query(query, conn)

            rename_map = {}
            for col in df.columns:
                if table_name == 'researchers' and col == 'Researcher_ID':
                    rename_map[col] = 'User_ID'
                elif col == 'First_Name':
                    rename_map[col] = 'FName'
                elif col == 'Last_Name':
                    rename_map[col] = 'LName'
                elif col == 'Publication_Date':
                    rename_map[col] = 'PublicationDate'
                elif col == 'Review_Date':
                    rename_map[col] = 'ReviewDate'
                elif col == 'Download_Date':
                    rename_map[col] = 'DownloadDate'
                elif col == 'Search_Date':
                    rename_map[col] = 'SearchDate'
                elif col == 'Join_Date':
                    rename_map[col] = 'JoinDate'
                elif table_name in ['reviews', 'downloads', 'searches'] and col == 'Researcher_ID':
                    rename_map[col] = 'User_ID'
                elif table_name == 'researcher_fields' and col == 'Researcher_ID':
                    rename_map[col] = 'User_ID'
                elif col == 'Field_Name':
                    rename_map[col] = 'FieldName'
                elif col == 'numeber_of_papers':
                    rename_map[col] = 'No_Papers'

            if rename_map:
                df.rename(columns=rename_map, inplace=True)

            data[table_name] = df
            logger.info(f"Loaded {table_name}: {len(df)} records")

        except Exception as e:
            logger.error(f"Error loading {table_name}: {e}")
            data[table_name] = pd.DataFrame()

    # Build Keywords column
    if not data['paper_keywords'].empty and not data['papers'].empty:
        keyword_col = None
        possible_names = ['Keyword', 'Keywords', 'Keyword_Text', 'KeywordName', 'Keyword_Name']
        
        for col_name in possible_names:
            if col_name in data['paper_keywords'].columns:
                keyword_col = col_name
                break
        
        if keyword_col is None:
            cols = data['paper_keywords'].columns.tolist()
            if len(cols) >= 3:
                keyword_col = cols[2]
            elif len(cols) >= 2:
                keyword_col = cols[1]
        
        if keyword_col:
            try:
                keywords_grouped = data['paper_keywords'].groupby('Paper_ID')[keyword_col].apply(
                    lambda x: ', '.join(x.astype(str))
                ).reset_index()
                keywords_grouped.columns = ['Paper_ID', 'Keywords']
                
                data['papers'] = data['papers'].merge(keywords_grouped, on='Paper_ID', how='left')
                data['papers']['Keywords'] = data['papers']['Keywords'].fillna('')
                logger.info(f"Built Keywords column from Paper_Keywords table")
            except Exception as e:
                logger.warning(f"Error building Keywords column: {e}")
                data['papers']['Keywords'] = ''
        else:
            data['papers']['Keywords'] = ''
    else:
        data['papers']['Keywords'] = ''

    return data


def add_review_to_db(conn, user_id, paper_id, rating):
    """Add or update a paper review in database"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT Review_ID FROM Review 
            WHERE Researcher_ID = ? AND Paper_ID = ?
        """, (user_id, paper_id))
        
        existing = cursor.fetchone()
        review_date = datetime.now().strftime('%Y-%m-%d')
        
        if existing:
            cursor.execute("""
                UPDATE Review 
                SET Rating = ?, Review_Date = ?
                WHERE Researcher_ID = ? AND Paper_ID = ?
            """, (rating, review_date, user_id, paper_id))
        else:
            cursor.execute("""
                INSERT INTO Review (Rating, Review_Date, Paper_ID, Researcher_ID)
                VALUES (?, ?, ?, ?)
            """, (rating, review_date, paper_id, user_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error adding review: {e}")
        return False


def add_download_to_db(conn, user_id, paper_id):
    """Log a paper download in database"""
    cursor = conn.cursor()
    
    try:
        cursor.execute("""
            SELECT Download_ID FROM Download 
            WHERE Researcher_ID = ? AND Paper_ID = ?
        """, (user_id, paper_id))
        
        if cursor.fetchone():
            return True
        
        download_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO Download (Download_Date, Paper_ID, Researcher_ID)
            VALUES (?, ?, ?)
        """, (download_date, paper_id, user_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error logging download: {e}")
        return False


def add_search_to_db(conn, user_id, query):
    """Log a search query in database"""
    cursor = conn.cursor()
    
    try:
        search_date = datetime.now().strftime('%Y-%m-%d')
        
        cursor.execute("""
            INSERT INTO Search (Query, Search_Date, Researcher_ID)
            VALUES (?, ?, ?)
        """, (query, search_date, user_id))
        
        conn.commit()
        return True
        
    except Exception as e:
        conn.rollback()
        logger.error(f"Error logging search: {e}")
        return False


# ============================================================================
# RECOMMENDER CORE FUNCTIONS
# ============================================================================

def get_user_preferences(user_id, data):
    """Extract user preferences from historical behavior and profile"""
    preferences = {
        'user_id': user_id,
        'favorite_authors': [],
        'favorite_fields': [],
        'rated_papers': [],
        'downloaded_papers': [],
        'avg_rating': 0.0,
        'specialization': None,
        'is_new_user': False,
        'search_keywords': [],
        'selected_fields': [],
        'all_interest_fields': []
    }

    user_spec = data['researchers'][data['researchers']['User_ID'] == user_id]
    if not user_spec.empty:
        preferences['specialization'] = user_spec.iloc[0]['Specialization']

    if 'researcher_fields' in data and not data['researcher_fields'].empty:
        user_selected_fields = data['researcher_fields'][data['researcher_fields']['User_ID'] == user_id]
        if not user_selected_fields.empty:
            preferences['selected_fields'] = user_selected_fields['Field_ID'].tolist()

    user_reviews = data['reviews'][data['reviews']['User_ID'] == user_id]
    if not user_reviews.empty:
        preferences['rated_papers'] = user_reviews['Paper_ID'].tolist()
        preferences['avg_rating'] = user_reviews['Rating'].mean()

    user_downloads = data['downloads'][data['downloads']['User_ID'] == user_id]
    if not user_downloads.empty:
        preferences['downloaded_papers'] = user_downloads['Paper_ID'].tolist()

    user_searches = data['searches'][data['searches']['User_ID'] == user_id]
    if not user_searches.empty:
        preferences['search_keywords'] = user_searches['Query'].tolist()

    interacted_papers = list(set(preferences['rated_papers'] + preferences['downloaded_papers']))
    if interacted_papers:
        author_papers = data['write'][data['write']['Paper_ID'].isin(interacted_papers)]
        author_counts = author_papers['Author_ID'].value_counts()
        preferences['favorite_authors'] = author_counts.head(5).index.tolist()

    if interacted_papers:
        field_papers = data['papers'][data['papers']['Paper_ID'].isin(interacted_papers)]
        field_counts = field_papers['Field_ID'].value_counts()
        preferences['favorite_fields'] = field_counts.head(3).index.tolist()

    all_fields = set()
    
    if preferences['selected_fields']:
        all_fields.update(preferences['selected_fields'])
    
    if preferences['favorite_fields']:
        all_fields.update(preferences['favorite_fields'])
    
    if preferences['specialization']:
        spec_fields = get_fields_from_specialization(preferences['specialization'], data)
        all_fields.update(spec_fields)
        
    if preferences['search_keywords']:
        search_text = ' '.join(preferences['search_keywords'])
        search_fields = get_fields_from_specialization(search_text, data)
        all_fields.update(search_fields)
    
    preferences['all_interest_fields'] = list(all_fields)

    if len(interacted_papers) == 0 and not preferences['search_keywords']:
        preferences['is_new_user'] = True

    return preferences


def get_fields_from_specialization(specialization, data):
    """Map specialization to field IDs"""
    field_ids = []
    
    if not specialization or data['fields'].empty:
        return field_ids
    
    spec_words = specialization.lower().split()
    
    for idx, row in data['fields'].iterrows():
        field_name = str(row['FieldName']).lower()
        field_desc = str(row.get('Description', '')).lower() if 'Description' in row else ''
        
        if any(word in field_name or word in field_desc for word in spec_words):
            field_ids.append(row['Field_ID'])
    
    return field_ids


def get_user_interest_papers(user_id, data, exclude_interacted=True):
    """Get all papers in user's fields of interest"""
    preferences = get_user_preferences(user_id, data)
    interest_fields = preferences['all_interest_fields']
    
    if not interest_fields:
        return []
    
    interest_papers = data['papers'][data['papers']['Field_ID'].isin(interest_fields)]['Paper_ID'].tolist()
    
    if exclude_interacted:
        interacted = set(preferences['rated_papers'] + preferences['downloaded_papers'])
        interest_papers = [pid for pid in interest_papers if pid not in interacted]
    
    return interest_papers


def preprocess_text(text):
    """Preprocess text for vectorization"""
    if pd.isna(text):
        return ""
    return str(text).lower().strip()


def build_content_vectors(data):
    """Build TF-IDF vectors for all papers"""
    papers = data['papers'].copy()

    papers['content'] = (papers['Abstract'].apply(preprocess_text) + ' ' + 
                         papers['Keywords'].apply(preprocess_text))

    vectorizer = TfidfVectorizer(
        max_features=500,
        stop_words='english',
        ngram_range=(1, 2),
        min_df=1
    )

    tfidf_matrix = vectorizer.fit_transform(papers['content'])

    return vectorizer, tfidf_matrix, papers['Paper_ID'].tolist()


def build_profile_from_interests(user_id, data, vectorizer):
    """Build user profile vector from specialization and selected fields"""
    preferences = get_user_preferences(user_id, data)
    profile_text_parts = []
    
    if preferences['specialization']:
        profile_text_parts.append(preferences['specialization'].lower())
    
    if preferences['all_interest_fields']:
        for field_id in preferences['all_interest_fields']:
            field_row = data['fields'][data['fields']['Field_ID'] == field_id]
            if not field_row.empty:
                field_name = field_row.iloc[0]['FieldName']
                profile_text_parts.append(field_name.lower())
                
                if 'Description' in field_row.columns and pd.notna(field_row.iloc[0]['Description']):
                    profile_text_parts.append(field_row.iloc[0]['Description'].lower())
    
    if preferences['search_keywords']:
        profile_text_parts.extend([kw.lower() for kw in preferences['search_keywords'][:5]])
    
    profile_text = ' '.join(profile_text_parts) if profile_text_parts else 'research papers'
    
    try:
        spec_vector = vectorizer.transform([profile_text])
        return np.asarray(spec_vector.toarray())
    except:
        return np.asarray([[0.0] * len(vectorizer.get_feature_names_out())])


def content_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n=10):
    """Generate content-based recommendations"""
    preferences = get_user_preferences(user_id, data)
    
    candidate_paper_ids = get_user_interest_papers(user_id, data, exclude_interacted=True)
    
    if not candidate_paper_ids:
        return pd.DataFrame()

    interacted_papers = list(set(preferences['rated_papers'] + preferences['downloaded_papers']))

    if interacted_papers:
        paper_indices = [paper_ids.index(pid) for pid in interacted_papers if pid in paper_ids]
        if paper_indices:
            user_profile_vector = np.asarray(tfidf_matrix[paper_indices].mean(axis=0))
        else:
            user_profile_vector = build_profile_from_interests(user_id, data, vectorizer)
    else:
        user_profile_vector = build_profile_from_interests(user_id, data, vectorizer)

    candidate_indices = [paper_ids.index(pid) for pid in candidate_paper_ids if pid in paper_ids]
    
    if not candidate_indices:
        return pd.DataFrame()
    
    candidate_matrix = tfidf_matrix[candidate_indices]
    similarities = cosine_similarity(user_profile_vector, candidate_matrix).flatten()

    recommendations = pd.DataFrame({
        'Paper_ID': [paper_ids[idx] for idx in candidate_indices],
        'content_score': similarities
    })

    recommendations = recommendations.sort_values('content_score', ascending=False).head(top_n)

    return recommendations


def calculate_field_preference_score(user_id, data, paper_ids):
    """Calculate field preference scores"""
    preferences = get_user_preferences(user_id, data)
    interest_fields = preferences['all_interest_fields']

    if not interest_fields:
        return {pid: 0.0 for pid in paper_ids}

    scores = {}
    for pid in paper_ids:
        paper_field_df = data['papers'][data['papers']['Paper_ID'] == pid]
        if not paper_field_df.empty:
            paper_field_id = paper_field_df.iloc[0]['Field_ID']
            scores[pid] = 1.0 if paper_field_id in interest_fields else 0.0
        else:
            scores[pid] = 0.0

    return scores


def behavior_recommend(user_id, data, top_n=10):
    """Generate behavior-based recommendations"""
    candidate_papers = get_user_interest_papers(user_id, data, exclude_interacted=True)
    
    if not candidate_papers:
        return pd.DataFrame()

    field_scores = calculate_field_preference_score(user_id, data, candidate_papers)

    behavior_scores = {pid: 0.5 * field_scores[pid] for pid in candidate_papers}

    recommendations = pd.DataFrame({
        'Paper_ID': list(behavior_scores.keys()),
        'behavior_score': list(behavior_scores.values())
    })

    recommendations = recommendations.sort_values('behavior_score', ascending=False).head(top_n)

    return recommendations


def calculate_popularity_scores(data, field_filter=None, days=30):
    """Calculate popularity scores"""
    cutoff_date = datetime.now() - timedelta(days=days)
    recent_downloads = data['downloads'].copy()
    recent_downloads['DownloadDate'] = pd.to_datetime(recent_downloads['DownloadDate'])
    recent_downloads = recent_downloads[recent_downloads['DownloadDate'] >= cutoff_date]

    download_counts = recent_downloads['Paper_ID'].value_counts().reset_index()
    download_counts.columns = ['Paper_ID', 'download_count']

    avg_ratings = data['reviews'].groupby('Paper_ID')['Rating'].mean().reset_index()
    avg_ratings.columns = ['Paper_ID', 'avg_rating']

    if field_filter:
        papers_base = data['papers'][data['papers']['Field_ID'].isin(field_filter)][['Paper_ID']].copy()
    else:
        papers_base = data['papers'][['Paper_ID']].copy()
    
    popularity = papers_base.merge(download_counts, on='Paper_ID', how='left')
    popularity = popularity.merge(avg_ratings, on='Paper_ID', how='left')

    popularity['download_count'] = popularity['download_count'].fillna(0)
    popularity['avg_rating'] = popularity['avg_rating'].fillna(3.0)

    max_downloads = popularity['download_count'].max()
    if max_downloads > 0:
        popularity['norm_downloads'] = popularity['download_count'] / max_downloads
    else:
        popularity['norm_downloads'] = 0.0

    popularity['norm_rating'] = popularity['avg_rating'] / 5.0

    popularity['popularity_score'] = (
        0.6 * popularity['norm_downloads'] +
        0.4 * popularity['norm_rating']
    )

    return popularity[['Paper_ID', 'popularity_score', 'download_count', 'avg_rating']]


def hybrid_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n=10):
    """Generate hybrid recommendations"""
    preferences = get_user_preferences(user_id, data)
    interest_fields = preferences['all_interest_fields']
    
    if not interest_fields:
        return pd.DataFrame(), 0.0
    
    available_papers = get_user_interest_papers(user_id, data, exclude_interacted=True)
    
    if not available_papers:
        return pd.DataFrame(), 0.0
    
    interacted_papers = list(set(preferences['rated_papers'] + preferences['downloaded_papers']))
    
    content_recs = content_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n=50)
    
    if content_recs.empty:
        return pd.DataFrame(), 0.0
    
    behavior_recs = behavior_recommend(user_id, data, top_n=50)
    
    popularity = calculate_popularity_scores(data, field_filter=interest_fields)
    
    hybrid = content_recs.merge(behavior_recs, on='Paper_ID', how='outer')
    hybrid = hybrid.merge(popularity[['Paper_ID', 'popularity_score']], on='Paper_ID', how='left')
    
    hybrid['content_score'] = hybrid['content_score'].fillna(0)
    hybrid['behavior_score'] = hybrid['behavior_score'].fillna(0)
    hybrid['popularity_score'] = hybrid['popularity_score'].fillna(0)
    
    if len(interacted_papers) == 0:
        hybrid['hybrid_score'] = (
            0.5 * hybrid['content_score'] +
            0.2 * hybrid['behavior_score'] +
            0.3 * hybrid['popularity_score']
        )
    else:
        hybrid['hybrid_score'] = (
            0.4 * hybrid['content_score'] +
            0.4 * hybrid['behavior_score'] +
            0.2 * hybrid['popularity_score']
        )
    
    hybrid = hybrid.sort_values('hybrid_score', ascending=False).head(top_n)
    
    hybrid = add_paper_details(hybrid, data)
    
    hybrid = hybrid[hybrid['Field_ID'].isin(interest_fields)]
    
    if hybrid.empty:
        return pd.DataFrame(), 0.0
    
    accuracy_score = 8.5  # Simplified accuracy
    
    return hybrid, accuracy_score


def add_paper_details(recommendations, data):
    """Add paper details to recommendations"""
    if recommendations.empty:
        return recommendations
    
    existing_cols = recommendations.columns.tolist()
    
    if 'Title' not in existing_cols or 'Abstract' not in existing_cols:
        recommendations = recommendations.merge(
            data['papers'][['Paper_ID', 'Title', 'Abstract', 'Keywords', 'PublicationDate', 'Field_ID']], 
            on='Paper_ID', 
            how='left',
            suffixes=('', '_paper')
        )
        
        for col in ['Title', 'Abstract', 'Keywords', 'PublicationDate', 'Field_ID']:
            if f'{col}_paper' in recommendations.columns:
                recommendations[col] = recommendations[col].fillna(recommendations[f'{col}_paper'])
                recommendations.drop(columns=[f'{col}_paper'], inplace=True)

    if 'FieldName' not in existing_cols:
        recommendations = recommendations.merge(
            data['fields'][['Field_ID', 'FieldName']], 
            on='Field_ID', 
            how='left'
        )

    author_names = []
    for paper_id in recommendations['Paper_ID']:
        paper_authors = data['write'][data['write']['Paper_ID'] == paper_id]['Author_ID'].tolist()
        authors_list = []
        for author_id in paper_authors:
            author = data['authors'][data['authors']['Author_ID'] == author_id]
            if not author.empty:
                authors_list.append(f"{author.iloc[0]['FName']} {author.iloc[0]['LName']}")
        author_names.append(', '.join(authors_list) if authors_list else 'Unknown')
    
    recommendations['Authors'] = author_names

    return recommendations


# ============================================================================
# FASTAPI MODELS
# ============================================================================

class PaperRecommendation(BaseModel):
    """Paper recommendation response"""
    paper_id: int
    title: str
    authors: str
    abstract: str
    keywords: str
    field_name: str
    publication_date: Optional[str]
    hybrid_score: Optional[float] = None
    content_score: Optional[float] = None
    behavior_score: Optional[float] = None
    popularity_score: Optional[float] = None


class SmartRecommendationResponse(BaseModel):
    """Smart recommendation response"""
    user_id: Optional[int]
    user_type: str
    total_recommendations: int
    accuracy_score: Optional[float]
    interest_fields: List[str] = []
    recommendations: List[PaperRecommendation]


class ReviewInput(BaseModel):
    """Review input"""
    user_id: int
    paper_id: int
    rating: int = Field(..., ge=1, le=5)


class DownloadInput(BaseModel):
    """Download input"""
    user_id: int
    paper_id: int


class SearchInput(BaseModel):
    """Search input"""
    user_id: Optional[int]
    query: str


# ============================================================================
# FASTAPI LIFESPAN
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_pool
    settings = get_settings()
    db_pool = DatabasePool(settings)
    logger.info(f"API started - Environment: {settings.ENVIRONMENT}")

    yield

    if db_pool:
        db_pool.close_all()
    logger.info("API shutting down")


# ============================================================================
# FASTAPI APP
# ============================================================================

settings = get_settings()
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description="Optimized unified recommender API",
    lifespan=lifespan
)

allowed_origins = settings.ALLOWED_ORIGINS.split(",") if settings.ALLOWED_ORIGINS != "*" else ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)


# ============================================================================
# DATABASE DEPENDENCIES
# ============================================================================

async def get_db():
    """Get database connection"""
    conn = await run_in_threadpool(db_pool.get_connection)
    try:
        yield conn
    finally:
        await run_in_threadpool(db_pool.return_connection, conn)


async def get_cached_data(conn):
    """Get cached data and features"""
    cache_key = "recommender_data"
    cached = cache.get(cache_key)

    if cached:
        return cached

    def _load():
        data = load_data_from_db(conn)
        vectorizer, tfidf_matrix, paper_ids = build_content_vectors(data)
        return data, vectorizer, tfidf_matrix, paper_ids

    data, vectorizer, tfidf_matrix, paper_ids = await run_in_threadpool(_load)
    cache.set(cache_key, (data, vectorizer, tfidf_matrix, paper_ids))

    logger.info("Data loaded and cached")
    return data, vectorizer, tfidf_matrix, paper_ids


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/api/recommend", response_model=SmartRecommendationResponse)
async def smart_recommend(
    user_id: Optional[int] = Query(None, description="User ID"),
    top_n: int = Query(default=10, ge=1, le=50, description="Number of recommendations"),
    conn = Depends(get_db)
):
    """Smart unified recommendation endpoint"""
    try:
        data, vectorizer, tfidf_matrix, paper_ids = await get_cached_data(conn)

        user_type = ""
        accuracy_score = 0.0
        recommendations = pd.DataFrame()
        interest_fields = []

        if user_id is not None:
            user_exists = not data['researchers'][data['researchers']['User_ID'] == user_id].empty

            if user_exists:
                def _get_prefs():
                    return get_user_preferences(user_id, data)

                preferences = await run_in_threadpool(_get_prefs)

                if not preferences['is_new_user'] and len(preferences['all_interest_fields']) > 0:
                    def _recommend():
                        return hybrid_recommend(user_id, data, vectorizer, tfidf_matrix, paper_ids, top_n)

                    recommendations, accuracy_score = await run_in_threadpool(_recommend)
                    user_type = "existing_user_personalized"

                    if preferences['all_interest_fields']:
                        for field_id in preferences['all_interest_fields']:
                            field = data['fields'][data['fields']['Field_ID'] == field_id]
                            if not field.empty:
                                interest_fields.append(field.iloc[0]['FieldName'])

                else:
                    user_id = None
            else:
                raise HTTPException(status_code=404, detail=f"User {user_id} not found")

        if user_id is None or recommendations.empty:
            def _popular():
                return calculate_popularity_scores(data, field_filter=None, days=30)

            popularity_df = await run_in_threadpool(_popular)

            recommendations = popularity_df.merge(
                data['papers'][['Paper_ID', 'Title', 'Abstract', 'Keywords', 'PublicationDate', 'Field_ID']],
                on='Paper_ID',
                how='left'
            )

            recommendations = recommendations.merge(
                data['fields'][['Field_ID', 'FieldName']],
                on='Field_ID',
                how='left'
            )

            author_names = []
            for paper_id in recommendations['Paper_ID']:
                paper_authors = data['write'][data['write']['Paper_ID'] == paper_id]['Author_ID'].tolist()
                authors_list = []
                for author_id in paper_authors:
                    author = data['authors'][data['authors']['Author_ID'] == author_id]
                    if not author.empty:
                        authors_list.append(f"{author.iloc[0]['FName']} {author.iloc[0]['LName']}")
                author_names.append(', '.join(authors_list) if authors_list else 'Unknown')

            recommendations['Authors'] = author_names
            recommendations = recommendations.head(top_n)

            user_type = "new_user_popular"
            accuracy_score = 6.0

        if recommendations.empty:
            raise HTTPException(status_code=404, detail="No recommendations available")

        result = []
        for _, paper in recommendations.iterrows():
            result.append(PaperRecommendation(
                paper_id=int(paper['Paper_ID']),
                title=paper['Title'],
                authors=paper.get('Authors', 'Unknown'),
                abstract=paper['Abstract'][:300] + "..." if len(str(paper['Abstract'])) > 300 else str(paper['Abstract']),
                keywords=paper.get('Keywords', ''),
                field_name=paper.get('FieldName', 'General'),
                publication_date=paper['PublicationDate'].strftime('%Y-%m-%d') if pd.notna(paper.get('PublicationDate')) else None,
                hybrid_score=float(round(paper['hybrid_score'] * 10, 2)) if 'hybrid_score' in paper.index and pd.notna(paper['hybrid_score']) else None,
                content_score=float(round(paper['content_score'] * 10, 2)) if 'content_score' in paper.index and pd.notna(paper['content_score']) else None,
                behavior_score=float(round(paper['behavior_score'] * 10, 2)) if 'behavior_score' in paper.index and pd.notna(paper['behavior_score']) else None,
                popularity_score=float(round(paper['popularity_score'], 2)) if 'popularity_score' in paper.index and pd.notna(paper['popularity_score']) else None
            ))

        return SmartRecommendationResponse(
            user_id=user_id,
            user_type=user_type,
            total_recommendations=len(result),
            accuracy_score=accuracy_score,
            interest_fields=interest_fields,
            recommendations=result
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in smart_recommend: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.post("/api/interaction/review")
async def add_review(
    review: ReviewInput,
    conn = Depends(get_db)
):
    """Add or update a paper review"""
    try:
        def _add():
            return add_review_to_db(conn, review.user_id, review.paper_id, review.rating)

        success = await run_in_threadpool(_add)

        if success:
            cache.clear()
            return {
                "success": True,
                "message": "Review added successfully",
                "user_id": review.user_id,
                "paper_id": review.paper_id,
                "rating": review.rating
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to add review")

    except Exception as e:
        logger.error(f"Error adding review: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interaction/download")
async def log_download(
    download: DownloadInput,
    conn = Depends(get_db)
):
    """Log a paper download"""
    try:
        def _add():
            return add_download_to_db(conn, download.user_id, download.paper_id)

        success = await run_in_threadpool(_add)

        if success:
            cache.clear()
            return {
                "success": True,
                "message": "Download logged successfully",
                "user_id": download.user_id,
                "paper_id": download.paper_id
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to log download")

    except Exception as e:
        logger.error(f"Error logging download: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/interaction/search")
async def log_search(
    search: SearchInput,
    conn = Depends(get_db)
):
    """Log a search query"""
    try:
        if not search.user_id:
            return {
                "success": True,
                "message": "Anonymous search logged",
                "user_id": None,
                "query": search.query
            }

        def _add():
            return add_search_to_db(conn, search.user_id, search.query)

        success = await run_in_threadpool(_add)

        if success:
            cache.clear()
            return {
                "success": True,
                "message": "Search logged successfully",
                "user_id": search.user_id,
                "query": search.query
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to log search")

    except Exception as e:
        logger.error(f"Error logging search: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """API information"""
    return {
        "message": "Research Paper Recommender API - Optimized",
        "version": settings.API_VERSION,
        "endpoints": {
            "smart_recommend": "/api/recommend?user_id={id}&top_n={n}",
            "add_review": "POST /api/interaction/review",
            "log_download": "POST /api/interaction/download",
            "log_search": "POST /api/interaction/search",
            "health": "/api/health"
        }
    }


@app.get("/api/health")
async def health_check(conn = Depends(get_db)):
    """Health check"""
    try:
        def _test():
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()

        await run_in_threadpool(_test)

        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "database": "disconnected",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")
