# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an Enterprise RAG (Retrieval-Augmented Generation) system built with Python, LangChain, and Streamlit. The system provides intelligent document Q&A capabilities with a web interface.

## Architecture

### Core Components

**rag_system.py** - The main RAG system implementation
- `EnterpriseRAGSystem`: Main system class handling document ingestion, querying, and system management
- `DocumentProcessor`: Handles document loading and text splitting for multiple formats (PDF, DOCX, XLSX, TXT, CSV)
- `SmartCacheSystem`: SQLite-based query caching system
- Supports Ollama LLMs and HuggingFace embeddings
- Uses Chroma for vector storage with persistence

**streamlit_rag_app.py** - Web interface
- Comprehensive Streamlit application with document upload, chat interface, analytics dashboard
- Session state management for chat history and system status
- File upload handling with temporary storage
- Analytics with Plotly visualizations

**main.py** - Application entry point (identical to streamlit_rag_app.py)

### Key Architecture Patterns

- **Modular Design**: Core RAG logic separated from UI layer
- **Configuration-Based**: System behavior controlled via config dictionaries
- **Caching Layer**: Query results cached in SQLite for performance
- **Error Handling**: Comprehensive exception handling with logging
- **Persistence**: Vector stores persisted to disk in `./vectordb/` directory

## Development Commands

### Running the Application

```bash
# Run the Streamlit web interface
streamlit run streamlit_rag_app.py
# or
streamlit run main.py

# Test the RAG system directly
python rag_system.py
```

### System Requirements

- Python 3.10+
- Ollama installed and running (for LLM functionality)
- Required Python packages: streamlit, langchain, langchain-ollama, langchain-huggingface, langchain-community, plotly, pandas, curl-cffi, sqlite3

### Configuration

The system uses configuration dictionaries with these key parameters:
- `llm_type`: "ollama" 
- `llm_config`: Model name and temperature settings
- `embedding_type`: "huggingface" or "ollama"
- `embedding_model`: Model identifier
- `retrieval_k`: Number of documents to retrieve (default: 5)

## File Structure

```
ent_RAG/
├── rag_system.py           # Core RAG implementation
├── streamlit_rag_app.py    # Streamlit web interface
├── main.py                 # Application entry point
├── vectordb/               # Vector database storage (created at runtime)
├── temp_uploads/           # Temporary file storage (created at runtime)
├── rag_system.log          # System logs
├── query_logs.jsonl        # Query history logs
└── rag_cache.db           # SQLite cache database
```

## Working with the System

### Document Ingestion

Documents are processed through the `DocumentProcessor` class:
- Supports PDF, DOCX, XLSX, TXT, CSV formats
- Documents are split into chunks (default: 1000 chars, 200 overlap)
- Chunks are embedded and stored in Chroma vector database
- Each collection is stored in `./vectordb/{collection_name}/`

### Query Processing

The query flow:
1. Check cache for existing response
2. Retrieve relevant document chunks using similarity search
3. Build context from retrieved documents
4. Generate response using LLM with prompt template
5. Cache response and log query

### System Health and Monitoring

The system includes comprehensive monitoring:
- Health checks for all components (LLM, embeddings, vector store, cache)
- Query logging with response times
- System statistics (document count, query count, average response time)
- Cache management and clearing capabilities

## Common Tasks

### Adding New Document Formats

1. Add loader class to `DocumentProcessor.__init__()` loaders dictionary
2. Import the appropriate LangChain document loader
3. Test with sample documents

### Changing LLM Models

Update the configuration in `load_config()` or through the Streamlit interface:
```python
"llm_config": {
    "model": "your_model_name",
    "temperature": 0.1
}
```

### Debugging

- Check `rag_system.log` for system logs
- Review `query_logs.jsonl` for query history
- Use the health check endpoint: `rag_system.health_check()`
- Monitor the analytics dashboard in the Streamlit interface