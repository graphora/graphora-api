# Graphora API

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green)](https://fastapi.tiangolo.com/)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.0+-008CC1)](https://neo4j.com/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Demo](https://img.shields.io/badge/Demo-demo.graphora.io-orange)](https://demo.graphora.io)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/graphora/graphora-api/blob/main/examples/quickstart.ipynb)

> Transform documents into knowledge graphs with AI

## Quick Start (5 minutes)

Get started with Graphora in minutes, not hours. Choose your preferred option:

### Option 1: Try the Demo (30 seconds)
Visit [demo.graphora.io](https://demo.graphora.io) - no signup required. Upload a document and see the knowledge graph extraction in action.

### Option 2: Google Colab (2 minutes)
Open our quickstart notebook and run it in your browser:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/graphora/graphora-api/blob/main/examples/quickstart.ipynb)

### Option 3: CLI (5 minutes)
Extract knowledge graphs from the command line:

```bash
# Install
pip install graphora[cli]

# Extract (uses the hosted demo at demo.graphora.io — no database or API key needed)
graphora extract document.pdf --output graph.json
```

That's it! No database setup, and no LLM keys of your own required to try the hosted demo.

> **Want to self-host?** Then you will need an LLM key. Jump to [Self-Hosting](#self-hosting) below — the [Zero-Config Mode](#zero-config-mode) section is explicit about which key to set (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, or `GEMINI_API_KEY`). Work is underway to add an Ollama-auto-detect path that removes this requirement for local runs — see [[product-strategy]] in the work vault.

## Why Graphora?

| Feature | Graphora | LangChain GraphTransformer | Microsoft GraphRAG |
|---------|----------|---------------------------|-------------------|
| Zero-config start | ✅ Yes | ⚠️ Partial | ❌ No |
| Auto schema inference | ✅ Yes | ❌ No | ❌ No |
| Quality validation | ✅ Yes | ❌ No | ❌ No |
| Human review workflow | ✅ Yes | ❌ No | ❌ No |
| Visual schema builder | ✅ Yes | ❌ No | ❌ No |
| Schema chat copilot | ✅ Yes | ❌ No | ❌ No |
| Entity deduplication | ✅ Yes (Splink) | ⚠️ Partial | ✅ Yes |

## Features

- **AI-powered extraction**: Advanced LLM-driven entity and relationship extraction from unstructured documents
- **Multi-format support**: Process PDFs, plain text (`.txt`, `.md`, `.csv`, `.json`, `.xml`, `.html`), Office formats (`.docx`, `.xlsx`, `.pptx`) via MarkItDown, and URLs via trafilatura. Install extras `graphora-server[pdf,docling,url]` for the full document-type surface
- **Visual schema builder**: Design your ontology with an intuitive drag-and-drop interface
- **Schema chat copilot**: Natural language conversations with streaming responses to refine your schema
- **Auto schema inference**: Let AI suggest schemas from your documents
- **Entity deduplication**: Powered by Splink for accurate entity resolution
- **Human-in-the-loop**: Review and refine extractions before final graph integration
- **Quality validation**: Built-in validation to ensure extraction completeness and accuracy
- **Flexible storage**: In-memory mode for quick starts, Neo4j for production
- **Real-time tracking**: Monitor preprocessing and extraction progress
- **Scalable architecture**: Microservice design optimized for document intelligence

## Self-Hosting

Want to run Graphora on your infrastructure? Here's how to get started.

### Prerequisites

- Python 3.11 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Quick Start

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
git clone https://github.com/graphora/graphora-api.git
cd graphora-api
uv sync

# Start development server
make dev
```

The API will be available at:
- API: http://localhost:8000
- Documentation: http://localhost:8000/api/v1/docs
- OpenAPI Spec: http://localhost:8000/api/v1/openapi.json

### Zero-Config Mode

Perfect for local development and testing:

```bash
# Set environment variables
export AUTH_BYPASS_ENABLED=true
export STORAGE_TYPE=memory
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key-here

# Start server
make dev
```

No database setup required. The system will use in-memory storage and auto-infer schemas.

### Full Setup with Neo4j

For production deployments:

```bash
# Start all services (Postgres, Neo4j, Prefect, Redis)
make compose-up

# Run migrations
make migrate

# Start server
make dev
```

See [Local Development Guide](docs/LOCAL_DEVELOPMENT.md) for detailed setup instructions.

## Developer Shortcuts

**Setup:**
- `make install` - sync Python dependencies via uv
- `make install-dev` - install with dev dependencies

**Development:**
- `make dev` - start development server with auto-reload
- `make start` - start production server

**Local Services:**
- `make compose-up` - start Postgres, Neo4j, Prefect, and Redis containers
- `make compose-down` - stop local services
- `make compose-logs` - tail logs from local services
- `make compose-status` - show status of local services

**Testing:**
- `make test` - run all tests (emits coverage stats and writes `coverage.xml`)
- `make test-unit` - run unit tests only
- `make test-integration` - run integration tests only
- `make test-cov` - run tests with HTML coverage report

**Code Quality:**
- `make lint` - run Ruff + Black checks
- `make lint-fix` - run Ruff + Black with auto-fix
- `make format` - apply Black formatting
- `make typecheck` - run mypy type checking
- `make deadcode` - run Vulture to surface unused definitions
- `make pre-commit` - run all pre-commit checks (lint, test, deadcode)

**Database:**
- `make migrate` - run database migrations
- `make dev-reset-postgres` - delete local Postgres data
- `make dev-reset-neo4j` - delete local Neo4j data
- `make dev-reset-redis` - delete local Redis data

**Other:**
- `make openapi-snapshot` - regenerate `tests/snapshots/openapi.json`
- `make clean` - remove build artifacts and cache files
- `make help` - show all available commands

## Authentication

All API requests must include a Clerk-issued bearer token:

```bash
Authorization: Bearer <token>
```

Configure the backend with Clerk credentials via `.env`:
- `CLERK_JWKS_URL`
- `CLERK_ISSUER`
- `CLERK_AUDIENCE`
- `CLERK_API_KEY` (if server-to-server calls are required)

Clients no longer send the legacy `user-id` header; the backend derives the user from the JWT subject claim.

### Service-to-service / pipeline tokens

Call the API from CI jobs or data pipelines without extra backend code. Mint short-lived Clerk JWTs on demand:

1. Create (or reuse) a Clerk user that represents the pipeline and add a JWT template (e.g. `graphora_pipeline`) whose `aud` value matches `CLERK_AUDIENCE`.

2. When the pipeline starts, create a token via Clerk's backend API:
   ```bash
   curl -X POST "https://api.clerk.com/v1/users/<USER_ID>/tokens/graphora_pipeline" \
     -H "Authorization: Bearer $CLERK_API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"expires_in_seconds": 3600}'
   ```

3. Export the returned token before invoking the client:
   ```bash
   export GRAPHORA_AUTH_TOKEN="<clerk-jwt-from-step-2>"
   python pipeline.py
   ```

4. Repeat the minting step whenever the token expires (keep TTLs short and rotate the Clerk API key like any other secret).

The `graphora` Python package automatically reads `GRAPHORA_AUTH_TOKEN`, so no application changes are required.

## Project Structure

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
   - Integrates with LLM providers for intelligent processing
   - Handles temporary graph creation

3. **Graph Service**
   - Manages Neo4j database operations
   - Handles subgraph creation and updates
   - Processes user feedback

## API Endpoints

### REST API

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

See the [API Documentation](http://localhost:8000/api/v1/docs) for the complete endpoint reference.

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

## Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Before Contributing

1. Read the [Code of Conduct](CODE_OF_CONDUCT.md)
2. Sign the [Contributor License Agreement](CLA.md)
3. Check out [good first issues](https://github.com/graphora/graphora-api/labels/good%20first%20issue)

## License

This project is licensed under the **MIT License**.

- ✅ Use freely in personal and commercial projects
- ✅ Modify and distribute with or without source code
- ✅ Use in closed-source and SaaS products

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
