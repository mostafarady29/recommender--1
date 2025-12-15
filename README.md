# Python Recommender System

Research Paper Recommender System using Hybrid Recommendation (Content + Behavior + Popularity)

## Features
- Smart unified recommendation endpoint
- User preference learning
- Field-based filtering
- Interaction tracking
- Auto-deployed on Railway

## API Endpoints

- `GET /api/recommend?user_id={id}&top_n={n}` - Get recommendations
- `POST /api/interaction/review` - Log paper review
- `POST /api/interaction/download` - Log paper download
- `POST /api/interaction/search` - Log search query
- `GET /api/health` - Health check

## Deployment

This service is deployed on Railway and connects to Azure SQL Database.

### Environment Variables Required:
```env
DB_NAME=Insight
DB_USER=your_username
DB_PASSWORD=your_password
DB_HOST=yourserver.database.windows.net
DB_PORT=1433
DB_DRIVER=ODBC Driver 18 for SQL Server
ALLOWED_ORIGINS=https://your-frontend.vercel.app
```

## Local Development

```bash
pip install -r requirements.txt
uvicorn recommender_api:app --reload
```

## Documentation

API docs available at `/docs` (development only)
