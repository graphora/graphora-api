# graphit-api

A sophisticated document processing backend that leverages AI-powered intelligent document analysis with flexible and modular preprocessing capabilities. The system uses FastAPI, LLM, and Neo4j to create an intelligent document processing pipeline.

## Features

- Advanced AI-driven document processing pipeline with robust error handling
- Multi-format document support and intelligent workflow management
- Scalable microservice architecture optimized for document intelligence
- Real-time status tracking for preprocessing steps
- Temporary subgraph creation for review before final integration
- Human-in-the-loop feedback system

## Prerequisites

- Python 3.11 or higher
- LLM API key
- Neo4j database access

## Getting Started

### Environment Setup

1. Fork this project on Replit
2. Add the following secrets to your Replit project:
   - `ANTHROPIC_API_KEY`: Your Anthropic API key `default` (or)
      `OPENAI_API_KEY`: Your OpenAI API key (or)
      `GOOGLE_GEMINI_API_KEY`: Your Google Gemini AI Studio (or)
   - `NEO4J_URI`: Neo4j database URI
   - `NEO4J_USER`: Neo4j database username
   - `NEO4J_PASSWORD`: Neo4j database password

### Building the Project
- Install uv: 
  ````bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ````
- Create virtual env: 
  ````bash
  uv venv
  ````
- ```bash
   uv pip install -r requirements.txt
   uv pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-3.7.1/en_core_web_sm-3.7.1.tar.gz
   ```



### Running the Project

The project will automatically start when you run it on Replit. The FastAPI server will be available at port 8000.

To manually start the server:

```bash
python -m app.main
```

The API will be available at:
- API Documentation: `/api/v1/docs`
- OpenAPI Specification: `/api/v1/openapi.json`

## Development Guide

### Project Structure

```
app/
├── agents/             # AI agents for workflow and feedback
├── api/               # API endpoints (REST and GraphQL)
├── services/          # Core business logic services
├── schemas/           # Pydantic models and schemas
├── utils/             # Utility functions and helpers
└── main.py           # Application entry point
```

### Key Components

1. **Preprocessing Service**
   - Handles multi-step document preprocessing
   - Provides real-time status updates
   - Implements robust error handling

2. **Extraction Service**
   - Manages entity and relationship extraction
   - Integrates with OpenAI for intelligent processing
   - Handles temporary graph creation

3. **Graph Service**
   - Manages Neo4j database operations
   - Handles subgraph creation and updates
   - Processes user feedback

### API Endpoints

#### REST API

1. Document Upload
```http
POST /api/v1/documents/upload
Content-Type: multipart/form-data
```

2. Submit Feedback
```http
POST /api/v1/feedback/{document_id}
Content-Type: application/json
```

## Error Handling

The system implements comprehensive error handling:
- Status tracking for each preprocessing step
- Detailed error messages and logging
- Graceful failure recovery
- User-friendly error responses

## Deployment

The project is configured to run automatically with the following features:
- Auto-reload during development
- Production-ready ASGI server (uvicorn)
- Proper port configuration for Replit hosting

### Production Considerations

When deploying to production:
1. Update CORS settings in `main.py`
2. Configure proper logging levels
3. Set up proper database credentials
4. Enable rate limiting and security measures

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
