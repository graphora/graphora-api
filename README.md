# Graphora API

[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-008CC1)](https://neo4j.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

> AI-powered knowledge graph construction from unstructured data

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

### Building the Project
- Install uv: 
  ````bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ````
- Create virtual env: 
  ````bash
  uv venv
  ````
- Install `libmagic`
  ````bash
  sudo apt-get install libmagic1
  ````
- ```bash
   uv sync
   ```
- [Setup BAML](https://docs.boundaryml.com/guide/installation-language/python)

### Developer Shortcuts

- `make install` – sync Python dependencies via uv
- `uv sync --group dev` – install optional dev tools (e.g. Vulture for dead-code checks)
- `make compose-up` – start local Neo4j and Redis containers (see [Local Development Guide](docs/LOCAL_DEVELOPMENT.md))
- `make lint`, `make test`, `make typecheck` – run quality gates before committing
- `make test-unit`, `make test-integration` – run just unit or integration slices as needed
- `make test` now emits coverage stats to the terminal and writes `coverage.xml` for CI tooling
- `make deadcode` – run Vulture against the codebase to surface unused definitions
- `make openapi-snapshot` – regenerate `tests/snapshots/openapi.json` after intentional API changes so contract tests stay green



### Running the Project

The project will automatically start when you run it on Replit. The FastAPI server will be available at port 8000.

To manually start the server:

```bash
python -m app.main
```

The API will be available at:
- API Documentation: `/api/v1/docs`
- OpenAPI Specification: `/api/v1/openapi.json`

### Authentication
- All API requests must include a Clerk-issued bearer token: `Authorization: Bearer <token>`.
- Configure the backend with Clerk credentials via `.env`: `CLERK_JWKS_URL`, `CLERK_ISSUER`, `CLERK_AUDIENCE`, and `CLERK_API_KEY` if server-to-server calls are required.
- Clients no longer send the legacy `user-id` header; the backend derives the user from the JWT subject claim.

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

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Before Contributing

1. Read the [Code of Conduct](CODE_OF_CONDUCT.md)
2. Sign the [Contributor License Agreement](CLA.md)
3. Check out [good first issues](https://github.com/graphora/graphora-api/labels/good%20first%20issue)

## Documentation

- [Contributing Guide](CONTRIBUTING.md) - How to contribute
- [Repository Guidelines](AGENTS.md) - Quick contributor reference
- [Local Development Guide](docs/LOCAL_DEVELOPMENT.md) - Spin up dependencies and run the API locally
- [Security Policy](SECURITY.md) - How to report security issues
- [Support](SUPPORT.md) - How to get help
- [Trademark Policy](TRADEMARK.md) - Trademark usage guidelines

## Related Repositories

- **Frontend**: [graphora/graphora-fe](https://github.com/graphora/graphora-fe)
- **Python Client**: [graphora/graphora-client](https://github.com/graphora/graphora-client)

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

- ✅ Use for free under AGPL v3 terms
- ✅ Modify and distribute with source code
- ❌ Cannot use as closed-source SaaS without commercial license

For commercial licensing (closed-source SaaS, enterprise deployments, OEM), contact: **sales@graphora.io**

See [LICENSE](LICENSE) for full terms.

## Commercial Support

- **Enterprise Support**: SLA-backed support for production deployments
- **Consulting**: Custom integrations, training, architecture design
- **Commercial Licensing**: Closed-source and SaaS deployments
- **Database Vendor Partnerships**: OEM licensing for database companies

Contact: **support@graphora.io**

## Community

- **GitHub Discussions**: [Ask questions, share ideas](https://github.com/graphora/graphora-api/discussions)
- **Discord**: Coming soon
- **Twitter**: Coming soon

## Security

Please report security vulnerabilities to **support@graphora.io**

See [SECURITY.md](SECURITY.md) for details.

---

Made with ❤️ by [Arivan Labs](https://arivanlabs.com)
