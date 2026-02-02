# Graphora Adoption Improvement Plan

## Overview

This plan addresses key adoption barriers identified through market research and product analysis. The goal is to reduce time-to-first-graph from 2-4 hours to under 5 minutes.

**Existing Assets:**
- Hosted demo: https://demo.graphora.io
- Visual ontology builder (Frontend)
- Schema generation copilot (Frontend)

---

## Phase 1: Remove Friction (Week 1-2)

### 1.1 Remove Marker API & libmagic Dependencies ✅ COMPLETE

**Problem:** Documentation mentions Marker API for PDF processing and libmagic, but they're not actually used.

**Status:** Completed. Removed all Marker API settings, python-magic dependency, libmagic references from documentation, and the app/services/marker/ directory.

**Backend Changes:**

| File | Action |
|------|--------|
| `README.md` | Remove libmagic installation instructions |
| `app/config.py` | Remove `PDF_PROCESSOR`, `MARKER_API_*` settings |
| `docs/LOCAL_DEVELOPMENT.md` | Remove Marker references |
| `pyproject.toml` | Remove `python-magic` if present |

**Verification:**
```bash
grep -r "marker" app/
grep -r "libmagic\|python-magic" .
```

---

### 1.2 Local Auth Bypass (Remove Clerk for Local Dev) ✅ COMPLETE

**Problem:** Clerk authentication adds friction for local development and testing.

**Status:** Completed. Added `AUTH_BYPASS_ENABLED` to backend and `NEXT_PUBLIC_AUTH_BYPASS` to frontend. When enabled, both skip Clerk authentication and use mock user context.

**Backend Changes (`graphora-api`):**

| File | Change |
|------|--------|
| `app/auth/dependencies.py` | Add `LOCAL_AUTH_BYPASS` mode |
| `app/config.py` | Add `AUTH_BYPASS_ENABLED: bool = False` |
| `.env.example` | Add `AUTH_BYPASS_ENABLED=true` for local |

**Implementation:**

```python
# app/auth/dependencies.py
async def get_current_user(request: Request) -> AuthenticatedUser:
    if settings.AUTH_BYPASS_ENABLED:
        return AuthenticatedUser(
            user_id="local-dev-user",
            email="dev@localhost",
            is_authenticated=True
        )
    # ... existing Clerk validation
```

**Frontend Changes (`graphora-fe`):**

| File | Change |
|------|--------|
| `middleware.ts` | Skip protection when `NEXT_PUBLIC_AUTH_BYPASS=true` |
| `lib/auth-utils.ts` | Return mock context in bypass mode |
| `app/layout.tsx` | Conditional ClerkProvider wrapper |
| `.env.local.example` | Add `NEXT_PUBLIC_AUTH_BYPASS=true` |

**Implementation:**

```typescript
// lib/auth-utils.ts
export async function getBackendAuthContext() {
  if (process.env.NEXT_PUBLIC_AUTH_BYPASS === 'true') {
    return {
      userId: 'local-dev-user',
      token: 'bypass-token',
    };
  }
  // ... existing Clerk logic
}

// middleware.ts
export default function middleware(request: NextRequest) {
  if (process.env.NEXT_PUBLIC_AUTH_BYPASS === 'true') {
    return NextResponse.next();
  }
  return clerkMiddleware()(request);
}
```

---

### 1.3 In-Memory Graph Mode (No Staging DB Required)

**Problem:** Users must provision 2 Neo4j databases before trying the product.

**Goal:** Add option to store extracted graphs in-memory for quick experimentation.

**Backend Changes:**

| File | Change |
|------|--------|
| `app/config.py` | Add `GRAPH_STORAGE_MODE: Literal["neo4j", "memory"] = "neo4j"` |
| `app/services/storage/interface.py` | Already abstracted - good |
| `app/services/storage/memory.py` | **NEW** - In-memory graph storage |
| `app/services/storage/__init__.py` | Add factory function |

**New File: `app/services/storage/memory.py`**

```python
"""In-memory graph storage for quick experimentation."""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from .interface import GraphStorageInterface
from .models import NodeData, RelationshipData


@dataclass
class InMemoryGraph:
    """Single graph stored in memory."""
    nodes: Dict[str, NodeData] = field(default_factory=dict)
    relationships: List[RelationshipData] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


class InMemoryStorage(GraphStorageInterface):
    """In-memory implementation for zero-config experimentation.

    Features:
    - No database required
    - Fast iteration for development
    - Graphs persist only for session lifetime
    - Export to JSON/Cypher for later import to Neo4j
    """

    def __init__(self):
        self._graphs: Dict[str, InMemoryGraph] = {}
        self._user_graphs: Dict[str, List[str]] = {}  # user_id -> graph_ids

    async def store_nodes(
        self,
        transform_id: str,
        nodes: List[NodeData],
        user_id: str,
    ) -> int:
        graph = self._get_or_create_graph(transform_id, user_id)
        for node in nodes:
            graph.nodes[node.id] = node
        return len(nodes)

    async def store_relationships(
        self,
        transform_id: str,
        relationships: List[RelationshipData],
        user_id: str,
    ) -> int:
        graph = self._get_or_create_graph(transform_id, user_id)
        graph.relationships.extend(relationships)
        return len(relationships)

    async def get_graph(
        self,
        transform_id: str,
        user_id: str,
    ) -> Optional[Dict[str, Any]]:
        graph = self._graphs.get(transform_id)
        if not graph:
            return None
        return {
            "nodes": list(graph.nodes.values()),
            "relationships": graph.relationships,
            "metadata": graph.metadata,
        }

    async def export_to_cypher(self, transform_id: str) -> str:
        """Export graph as Cypher statements for Neo4j import."""
        graph = self._graphs.get(transform_id)
        if not graph:
            return ""

        statements = []
        for node in graph.nodes.values():
            props = ", ".join(f"{k}: {repr(v)}" for k, v in node.properties.items())
            statements.append(f"CREATE (:{node.label} {{{props}}})")

        for rel in graph.relationships:
            statements.append(
                f"MATCH (a), (b) WHERE a.id = '{rel.source}' AND b.id = '{rel.target}' "
                f"CREATE (a)-[:{rel.type}]->(b)"
            )

        return "\n".join(statements)

    async def export_to_json(self, transform_id: str) -> Dict[str, Any]:
        """Export graph as JSON for portability."""
        return await self.get_graph(transform_id, "")

    def _get_or_create_graph(self, transform_id: str, user_id: str) -> InMemoryGraph:
        if transform_id not in self._graphs:
            self._graphs[transform_id] = InMemoryGraph()
            if user_id not in self._user_graphs:
                self._user_graphs[user_id] = []
            self._user_graphs[user_id].append(transform_id)
        return self._graphs[transform_id]
```

**Storage Factory:**

```python
# app/services/storage/__init__.py
from app.config import settings

def get_storage() -> GraphStorageInterface:
    if settings.GRAPH_STORAGE_MODE == "memory":
        from .memory import InMemoryStorage
        return InMemoryStorage()
    else:
        from .neo4j import Neo4jStorage
        return Neo4jStorage()
```

**API Changes:**

| Endpoint | Change |
|----------|--------|
| `POST /api/v1/config` | Make optional when `GRAPH_STORAGE_MODE=memory` |
| `GET /api/v1/graph/{id}/export` | **NEW** - Export in-memory graph |

---

### 1.4 On-the-Fly Schema Generation

**Problem:** Users must create ontology before uploading documents.

**Goal:** When no ontology provided, auto-generate schema from document content.

**Backend Changes:**

| File | Change |
|------|--------|
| `app/api/transform.py` | Make `ontology_id` optional |
| `app/services/transform/graph_transformer.py` | Add schema inference |
| `app/services/schema_inference.py` | **NEW** - Auto-schema generation |

**New File: `app/services/schema_inference.py`**

```python
"""Automatic schema inference from document content."""

from typing import Dict, Any, List
from app.services.llm.client import get_llm_client


SCHEMA_INFERENCE_PROMPT = """
Analyze the following text and extract a knowledge graph schema.

Text:
{text_sample}

Generate a YAML ontology that captures:
1. Main entity types mentioned (people, organizations, concepts, etc.)
2. Key relationships between entities
3. Important properties for each entity type

Output ONLY valid YAML in this format:
```yaml
version: "0.1.0"
entities:
  EntityName:
    properties:
      property_name:
        type: str
        description: "..."
relationships:
  RELATIONSHIP_TYPE:
    source: SourceEntity
    target: TargetEntity
```
"""


async def infer_schema_from_text(
    text_chunks: List[str],
    max_sample_chars: int = 10000,
) -> Dict[str, Any]:
    """Infer ontology schema from document text.

    Args:
        text_chunks: List of text chunks from documents
        max_sample_chars: Maximum characters to sample for inference

    Returns:
        Parsed ontology dictionary
    """
    # Sample text from chunks
    sample = ""
    for chunk in text_chunks:
        if len(sample) >= max_sample_chars:
            break
        sample += chunk[:max_sample_chars - len(sample)] + "\n\n"

    # Call LLM for schema inference
    client = get_llm_client()
    response = await client.generate(
        prompt=SCHEMA_INFERENCE_PROMPT.format(text_sample=sample),
        temperature=0.3,  # Lower temperature for consistent schema
    )

    # Parse YAML from response
    import yaml
    yaml_match = re.search(r'```yaml\n(.*?)```', response, re.DOTALL)
    if yaml_match:
        return yaml.safe_load(yaml_match.group(1))

    # Fallback: try parsing entire response as YAML
    return yaml.safe_load(response)
```

**Transform API Change:**

```python
# app/api/transform.py
@router.post("/{ontology_id}/upload")
@router.post("/upload")  # NEW: ontology-free endpoint
async def upload_documents(
    ontology_id: Optional[str] = None,  # Now optional
    files: List[UploadFile] = File(...),
    auto_schema: bool = Query(False, description="Auto-generate schema from content"),
    user: AuthenticatedUser = Depends(get_current_user),
):
    if not ontology_id and not auto_schema:
        raise HTTPException(
            status_code=400,
            detail="Either ontology_id or auto_schema=true required"
        )

    if auto_schema:
        # Parse documents first
        text_chunks = await parse_documents(files)
        # Infer schema
        inferred_schema = await infer_schema_from_text(text_chunks)
        # Create temporary ontology
        ontology_id = await create_temp_ontology(inferred_schema, user.user_id)

    # Continue with normal transform flow...
```

---

## Phase 2: Improve Schema Copilot (Week 2-3)

### 2.1 Convert Q&A Template to Freeflow Chat

**Problem:** Current schema copilot is rigid Q&A template - users can't have natural conversation.

**Current Flow:**
```
Question 1 → Answer → Question 2 → Answer → Question 3 → Generate
```

**Target Flow:**
```
User: "I want to build a knowledge graph for my company's HR data"
Assistant: "I can help with that! Tell me more about what HR data you have..."
User: "We have employee records, departments, and project assignments"
Assistant: "Great! Here's an initial schema... [shows preview]"
User: "Can you add a skills tracking entity?"
Assistant: "Sure! I've added Skills entity with these properties... [updates preview]"
```

**Frontend Changes (`graphora-fe`):**

| File | Change |
|------|--------|
| `lib/store/schema-chat-store.ts` | Remove `QUESTION_SETS`, add freeflow state |
| `components/schema-chat/question-input.tsx` | Convert to chat input |
| `app/schema-chat/page.tsx` | Simplify to chat interface |
| `app/api/v1/schema-chat/route.ts` | **NEW** - Streaming chat endpoint |

**New Store Design:**

```typescript
// lib/store/schema-chat-store.ts
interface SchemaChatState {
  // Chat state
  messages: ChatMessage[];
  isStreaming: boolean;

  // Schema state (live preview)
  currentSchema: string | null;  // YAML
  schemaVersion: number;

  // Session
  sessionId: string | null;

  // Actions
  sendMessage: (content: string) => Promise<void>;
  updateSchemaFromChat: (yaml: string) => void;
  exportSchema: () => string;
  resetChat: () => void;
}

interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  schemaUpdate?: string;  // If message includes schema change
  timestamp: Date;
}
```

**Backend Streaming Endpoint:**

```python
# app/api/chat.py
@router.post("/schema-chat/stream")
async def schema_chat_stream(
    request: SchemaChatRequest,
    user: AuthenticatedUser = Depends(get_current_user),
):
    """Streaming chat endpoint for schema generation."""

    async def generate():
        async for chunk in schema_chat_service.stream_response(
            session_id=request.session_id,
            message=request.message,
            current_schema=request.current_schema,
            user_id=user.user_id,
        ):
            yield f"data: {json.dumps(chunk)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream"
    )
```

**Schema Chat Service:**

```python
# app/services/schema_chat_service.py
SCHEMA_CHAT_SYSTEM_PROMPT = """
You are a helpful assistant that helps users design knowledge graph schemas.

Your role:
1. Understand the user's domain and data
2. Suggest entity types and relationships
3. Iteratively refine the schema based on feedback
4. Output schema updates in YAML format

When you make changes to the schema, wrap them in:
```schema
<yaml content>
```

Be conversational and helpful. Ask clarifying questions when needed.
"""

class SchemaChatService:
    async def stream_response(
        self,
        session_id: str,
        message: str,
        current_schema: Optional[str],
        user_id: str,
    ) -> AsyncGenerator[Dict[str, Any], None]:

        # Build conversation history
        history = await self._get_session_history(session_id)

        # Construct prompt with current schema context
        messages = [
            {"role": "system", "content": SCHEMA_CHAT_SYSTEM_PROMPT},
            *history,
            {"role": "user", "content": message},
        ]

        if current_schema:
            messages.insert(1, {
                "role": "system",
                "content": f"Current schema:\n```yaml\n{current_schema}\n```"
            })

        # Stream response
        async for chunk in self.llm_client.stream(messages):
            # Check if chunk contains schema update
            if "```schema" in chunk:
                schema_yaml = self._extract_schema(chunk)
                yield {"type": "schema_update", "content": schema_yaml}
            else:
                yield {"type": "text", "content": chunk}

        # Save to history
        await self._save_to_history(session_id, message, full_response)
```

**Frontend Chat Component:**

```typescript
// components/schema-chat/chat-interface.tsx
export function ChatInterface() {
  const { messages, isStreaming, currentSchema, sendMessage } = useSchemaChatStore();
  const [input, setInput] = useState('');

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    await sendMessage(input);
    setInput('');
  };

  return (
    <div className="flex h-full">
      {/* Chat Panel */}
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          {isStreaming && <TypingIndicator />}
        </div>

        <form onSubmit={handleSubmit} className="p-4 border-t">
          <div className="flex gap-2">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Describe your data or ask for changes..."
              disabled={isStreaming}
            />
            <Button type="submit" disabled={isStreaming || !input.trim()}>
              Send
            </Button>
          </div>
        </form>
      </div>

      {/* Live Schema Preview */}
      <div className="w-1/2 border-l">
        <SchemaPreview schema={currentSchema} />
      </div>
    </div>
  );
}
```

---

## Phase 3: Developer Experience (Week 3-4)

### 3.1 CLI Tool

**Goal:** `graphora extract document.pdf --output graph.json`

**New Package: `graphora-cli`**

```
graphora-cli/
├── pyproject.toml
├── src/
│   └── graphora_cli/
│       ├── __init__.py
│       ├── main.py          # Entry point
│       ├── commands/
│       │   ├── extract.py   # Extract command
│       │   ├── schema.py    # Schema commands
│       │   └── serve.py     # Local server
│       └── client.py        # API client
```

**CLI Commands:**

```bash
# Quick extraction (auto-schema, in-memory)
graphora extract document.pdf --output graph.json

# With custom schema
graphora extract document.pdf --schema ontology.yaml --output graph.json

# Generate schema from documents
graphora schema infer document.pdf --output schema.yaml

# Interactive schema chat
graphora schema chat

# Start local server (no auth, in-memory)
graphora serve --port 8000

# Export to Neo4j
graphora export graph.json --neo4j bolt://localhost:7687
```

**Implementation:**

```python
# src/graphora_cli/main.py
import typer
from rich.console import Console
from rich.progress import Progress

app = typer.Typer(
    name="graphora",
    help="Transform documents into knowledge graphs"
)
console = Console()


@app.command()
def extract(
    files: List[Path] = typer.Argument(..., help="Documents to process"),
    schema: Optional[Path] = typer.Option(None, "--schema", "-s", help="Ontology YAML file"),
    output: Path = typer.Option("graph.json", "--output", "-o", help="Output file"),
    format: str = typer.Option("json", "--format", "-f", help="Output format: json, cypher, csv"),
    api_url: str = typer.Option("http://localhost:8000", "--api", help="Graphora API URL"),
):
    """Extract knowledge graph from documents."""

    with Progress() as progress:
        task = progress.add_task("Processing...", total=100)

        # Upload and process
        client = GraphoraClient(api_url)

        if schema:
            ontology_id = client.upload_ontology(schema.read_text())
        else:
            ontology_id = None  # Auto-schema mode

        transform_id = client.upload_documents(
            files=[f.read_bytes() for f in files],
            ontology_id=ontology_id,
            auto_schema=schema is None,
        )

        # Poll for completion
        while True:
            status = client.get_status(transform_id)
            progress.update(task, completed=status.percentage)

            if status.is_complete:
                break
            if status.is_failed:
                console.print(f"[red]Error: {status.error}[/red]")
                raise typer.Exit(1)

            time.sleep(1)

        # Download result
        graph = client.get_graph(transform_id)

        if format == "json":
            output.write_text(json.dumps(graph, indent=2))
        elif format == "cypher":
            output.write_text(client.export_cypher(transform_id))

        console.print(f"[green]Graph saved to {output}[/green]")
        console.print(f"  Nodes: {len(graph['nodes'])}")
        console.print(f"  Relationships: {len(graph['edges'])}")


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p"),
    memory_mode: bool = typer.Option(True, "--memory/--neo4j"),
):
    """Start local Graphora server (no auth, in-memory storage)."""
    import subprocess

    env = {
        "AUTH_BYPASS_ENABLED": "true",
        "GRAPH_STORAGE_MODE": "memory" if memory_mode else "neo4j",
        "API_PORT": str(port),
    }

    console.print(f"[green]Starting Graphora on http://localhost:{port}[/green]")
    console.print("  Auth: disabled")
    console.print(f"  Storage: {'in-memory' if memory_mode else 'Neo4j'}")

    subprocess.run(
        ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", str(port)],
        env={**os.environ, **env}
    )


if __name__ == "__main__":
    app()
```

**pyproject.toml:**

```toml
[project]
name = "graphora-cli"
version = "0.1.0"
description = "CLI for Graphora knowledge graph extraction"
dependencies = [
    "typer>=0.9.0",
    "rich>=13.0.0",
    "httpx>=0.25.0",
]

[project.scripts]
graphora = "graphora_cli.main:app"
```

---

### 3.2 Google Colab Notebook

**File:** `examples/quickstart.ipynb`

```python
# Cell 1: Setup
"""
# Graphora Quickstart
Transform any document into a knowledge graph in 5 minutes.
"""

# Install
!pip install graphora-cli -q

# Cell 2: Upload Document
from google.colab import files
uploaded = files.upload()
doc_path = list(uploaded.keys())[0]
print(f"Uploaded: {doc_path}")

# Cell 3: Extract Knowledge Graph
!graphora extract "{doc_path}" --output graph.json

# Cell 4: Visualize
import json
import networkx as nx
import matplotlib.pyplot as plt

with open('graph.json') as f:
    graph = json.load(f)

G = nx.DiGraph()
for node in graph['nodes']:
    G.add_node(node['id'], label=node['label'], **node.get('properties', {}))
for edge in graph['edges']:
    G.add_edge(edge['source'], edge['target'], label=edge.get('label', ''))

plt.figure(figsize=(12, 8))
pos = nx.spring_layout(G, k=2)
nx.draw(G, pos, with_labels=True, node_color='lightblue',
        node_size=2000, font_size=10, arrows=True)
edge_labels = nx.get_edge_attributes(G, 'label')
nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
plt.title("Extracted Knowledge Graph")
plt.show()

# Cell 5: Query the Graph
print("Nodes by type:")
from collections import Counter
types = Counter(n['label'] for n in graph['nodes'])
for label, count in types.most_common():
    print(f"  {label}: {count}")

print(f"\nTotal: {len(graph['nodes'])} nodes, {len(graph['edges'])} relationships")

# Cell 6: Export Options
# Save as Cypher for Neo4j import
!graphora export graph.json --format cypher --output import.cypher
print("Cypher export saved to import.cypher")

# Download results
files.download('graph.json')
files.download('import.cypher')
```

---

### 3.3 5-Minute Quickstart Blog Post

**File:** `docs/blog/5-minute-quickstart.md`

```markdown
---
title: "Extract Knowledge Graphs in 5 Minutes"
date: 2024-02-02
author: "Graphora Team"
description: "Transform any document into a queryable knowledge graph with zero setup"
---

# Extract Knowledge Graphs in 5 Minutes

Turn your documents into connected knowledge without database setup,
authentication, or configuration files.

## Option 1: Try the Demo (30 seconds)

Visit [demo.graphora.io](https://demo.graphora.io) and:
1. Upload a PDF or paste text
2. Watch the AI extract entities and relationships
3. Explore the interactive graph

No signup required.

## Option 2: Google Colab (2 minutes)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/graphora/graphora-api/blob/main/examples/quickstart.ipynb)

Click the badge, upload your document, run the cells. Done.

## Option 3: CLI (5 minutes)

### Install

```bash
pip install graphora-cli
```

### Extract

```bash
graphora extract report.pdf --output graph.json
```

That's it. The CLI:
- Auto-detects document structure
- Generates a schema from your content
- Extracts entities and relationships
- Outputs a portable JSON graph

### Visualize

```bash
graphora viz graph.json
```

Opens an interactive graph viewer in your browser.

### Export to Neo4j

```bash
graphora export graph.json --neo4j bolt://localhost:7687
```

## What's Happening Under the Hood?

1. **Document Parsing**: PDFs, Word docs, and text files are converted to structured text
2. **Schema Inference**: AI analyzes your content and creates an appropriate ontology
3. **Entity Extraction**: LLM identifies people, organizations, concepts, and their properties
4. **Relationship Detection**: Connections between entities are discovered and typed
5. **Deduplication**: Same entities mentioned multiple times are merged

## Customizing the Schema

Want more control? Create a schema file:

```yaml
# schema.yaml
entities:
  Person:
    properties:
      name: { type: str, required: true }
      title: { type: str }
  Company:
    properties:
      name: { type: str, required: true }
relationships:
  WORKS_AT:
    source: Person
    target: Company
```

Then:

```bash
graphora extract report.pdf --schema schema.yaml --output graph.json
```

## Next Steps

- [Interactive Schema Designer](https://demo.graphora.io/schema) - Build schemas visually
- [API Documentation](https://docs.graphora.io/api) - Integrate into your applications
- [Self-Hosting Guide](https://docs.graphora.io/deploy) - Run Graphora on your infrastructure

## Need Help?

- [GitHub Discussions](https://github.com/graphora/graphora-api/discussions)
- [Discord Community](https://discord.gg/graphora)
- Email: support@graphora.io
```

---

## Phase 4: Documentation & Marketing (Week 4-5)

### 4.1 Update README.md

**Changes:**

1. Lead with demo link and Colab badge
2. Add "5-minute quickstart" section at top
3. Move technical setup details to separate docs
4. Add comparison table vs competitors

**New README Structure:**

```markdown
# Graphora

> Transform documents into knowledge graphs with AI

[![Demo](https://img.shields.io/badge/Try-Demo-blue)](https://demo.graphora.io)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](...)

## Quick Start (5 minutes)

### Option 1: CLI
```bash
pip install graphora-cli
graphora extract document.pdf --output graph.json
```

### Option 2: Python
```python
from graphora import extract
graph = extract("document.pdf")
```

### Option 3: API
```bash
curl -X POST https://api.graphora.io/extract \
  -F "file=@document.pdf"
```

## Why Graphora?

| Feature | Graphora | LangChain | GraphRAG |
|---------|----------|-----------|----------|
| Zero-config start | ✅ | ⚠️ | ❌ |
| Quality validation | ✅ | ❌ | ❌ |
| Human review workflow | ✅ | ❌ | ❌ |
| Visual schema builder | ✅ | ❌ | ❌ |

## [Full Documentation →](https://docs.graphora.io)
```

---

## Implementation Priority

| Task | Priority | Effort | Impact |
|------|----------|--------|--------|
| Remove Marker/libmagic refs | P0 | 1 day | Low friction |
| Local auth bypass | P0 | 2 days | High friction |
| In-memory storage | P0 | 3 days | High friction |
| On-the-fly schema | P1 | 3 days | High friction |
| Freeflow schema chat | P1 | 5 days | Medium friction |
| CLI tool | P1 | 4 days | High adoption |
| Colab notebook | P1 | 1 day | High adoption |
| Blog post | P2 | 1 day | Marketing |
| README update | P2 | 1 day | Marketing |

**Total Estimated Effort:** 3-4 weeks

---

## Success Metrics

| Metric | Current | Target (3 months) |
|--------|---------|-------------------|
| GitHub stars | ~0 | 500+ |
| Time to first graph | 2-4 hours | <5 minutes |
| Demo monthly visitors | Unknown | 1000+ |
| CLI downloads | 0 | 500+ |
| Discord members | 0 | 100+ |

---

## Files to Create/Modify Summary

### New Files

| Path | Description |
|------|-------------|
| `app/services/storage/memory.py` | In-memory graph storage |
| `app/services/schema_inference.py` | Auto-schema from documents |
| `app/services/schema_chat_service.py` | Freeflow chat backend |
| `graphora-cli/` | CLI package (new repo or monorepo) |
| `examples/quickstart.ipynb` | Colab notebook |
| `docs/blog/5-minute-quickstart.md` | Blog post |

### Modified Files (Backend)

| Path | Change |
|------|--------|
| `app/config.py` | Add `AUTH_BYPASS_ENABLED`, `GRAPH_STORAGE_MODE` |
| `app/auth/dependencies.py` | Add bypass logic |
| `app/api/transform.py` | Make ontology_id optional |
| `app/api/chat.py` | Add streaming schema chat |
| `app/services/storage/__init__.py` | Add storage factory |
| `README.md` | Rewrite for adoption |
| `.env.example` | Add new config options |

### Modified Files (Frontend)

| Path | Change |
|------|--------|
| `lib/store/schema-chat-store.ts` | Remove Q&A, add freeflow |
| `lib/auth-utils.ts` | Add bypass mode |
| `middleware.ts` | Conditional protection |
| `app/layout.tsx` | Conditional ClerkProvider |
| `app/schema-chat/page.tsx` | Simplify to chat UI |
| `components/schema-chat/*` | Rebuild for freeflow |
