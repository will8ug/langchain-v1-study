# AGENTS.md - Development Guidelines for LangChain v1 Study

This document provides guidelines for agentic coding agents working in this LangChain v1 study repository.

## Build, Lint, and Test Commands

### Environment Setup
```bash
# Install dependencies using uv
uv sync

# Activate virtual environment
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows
```

### Testing
```bash
# Run all tests
uv run pytest

# Run single test file
uv run pytest tests/test_multimodel_gemini.py

# Run specific test function
uv run pytest tests/test_multimodel_gemini.py::test_generate_image_writes_file

# Run tests with verbose output
uv run pytest -v

# Run tests with coverage
uv run pytest --cov=app
```

### Code Quality
```bash
# Run ruff linter
uv run ruff check .

# Auto-fix ruff issues
uv run ruff check --fix .

# Format code with ruff
uv run ruff format .
```

## Code Style Guidelines

### Import Organization
```python
# 1. Standard library imports first
import os
import base64
from pathlib import Path

# 2. Third-party imports
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
import pytest

# 3. Local application imports
from app.common.llm_configs import deepseek
```

### Formatting and Types
- **Formatter**: Use ruff for code formatting
- **Type checking**: Disabled in this project (python.analysis.typeCheckingMode: "off")
- **Line length**: Follow ruff defaults (typically 88 characters)
- **Indentation**: 4 spaces
- **Quotes**: Prefer double quotes for strings

### Naming Conventions
- **Files**: `snake_case.py` (e.g., `basic_gemini.py`, `multimodel_gemini.py`)
- **Functions**: `snake_case` (e.g., `generate_image()`)
- **Variables**: `snake_case` (e.g., `mock_response`, `output_path`)
- **Constants**: `UPPER_SNAKE_CASE` (e.g., `API_KEY`, `BASE_URL`)
- **Classes**: `PascalCase` (e.g., `LLMConfig`)

### Environment and Configuration
- **Always load environment variables**: `load_dotenv()` at module level
- **Use centralized configs**: Import from `app.common.llm_configs` when available
- **Environment template**: Reference `.env.example` for required variables
- **API keys**: Use `os.getenv("VARIABLE_NAME")` pattern

### Error Handling
```python
# Basic try/except for file operations
try:
    with open(filename, 'r') as f:
        content = f.read()
    return content
except Exception as e:
    return f"Error reading file: {str(e)}"

# Context managers for resource management
with open(output_path, "wb") as f:
    f.write(decoded_content)
```

### Function Structure
```python
def function_name(param1: str, param2: int) -> str:
    """Brief description of function purpose."""
    # Implementation
    return result

if __name__ == "__main__":
    # Demo or test code
    result = function_name("test", 123)
    print(result)
```

### Testing Patterns
- **Test files**: Name with `test_` prefix in `/tests/` directory
- **Fixtures**: Use `@pytest.fixture` for mock data and setup
- **Mocking**: Use `@patch` decorator for external API calls
- **File I/O**: Use `tmp_path` fixture for temporary directories
- **Assertions**: Test both success and error conditions

```python
@pytest.fixture
def mock_data():
    return {"key": "value"}

@patch("module.external_class")
def test_function(mock_external, mock_data, tmp_path):
    # Arrange
    mock_external.return_value = Mock()
    
    # Act
    result = function_under_test()
    
    # Assert
    assert result == expected
```

### LangChain Specific Patterns
- **Model instantiation**: Direct class instantiation with model names
- **Streaming**: Use `model.stream()` for real-time responses
- **Messages**: Use appropriate message classes (HumanMessage, etc.)
- **Tools**: Use `@tool` decorator for function tools
- **Agents**: Use `create_agent()` with proper middleware configuration

### File Organization
```
app/
├── common/           # Shared utilities and configurations
├── basic_*.py        # Basic examples for different providers
├── reasoning_*.py   # Reasoning content examples
├── multimodel_*.py   # Multi-modal examples
└── streaming_*.py    # Streaming examples
```

### Development Workflow
1. **Environment**: Always work in the virtual environment (`.venv/`)
2. **Dependencies**: Use `uv` for package management
3. **Testing**: Write tests for new functionality
4. **Code quality**: Run ruff before committing
5. **Documentation**: Update relevant documentation

### API Integration Guidelines
- **Multiple providers**: Support Gemini, DeepSeek, OpenAI, Qwen
- **Web search**: Use Tavily for web search capabilities
- **Structured output**: Use Pydantic models when needed
- **Multi-modal**: Handle text, images, and PDFs appropriately
- **Error handling**: Graceful degradation for API failures

### Security and Best Practices
- **API keys**: Never commit actual API keys to repository
- **Environment variables**: Use `.env` file for local development
- **File operations**: Use context managers and proper error handling
- **Input validation**: Validate user inputs when processing files
- **Resource cleanup**: Ensure proper cleanup of temporary resources

## Project-Specific Notes

- **Python version**: Requires Python 3.13+
- **Package manager**: Uses `uv` (modern Python package manager)
- **Type checking**: Intentionally disabled for rapid development
- **Focus**: Educational/experimental project for LangChain v1 features
- **Multi-provider**: Designed to work with multiple LLM providers

When working on this codebase, prioritize clarity, educational value, and proper testing while following the established patterns and conventions.