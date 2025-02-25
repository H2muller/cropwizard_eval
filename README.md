# CropWizard RAG Evaluation Tool
A Python-based tool for evaluating RAG (Retrieval Augmented Generation) performance using the Ragas framework. This tool helps assess the quality of RAG+LLM responses by measuring various metrics including context precision, context recall, answer relevancy, faithfulness, and factual correctness. This tool was developed as part of the CropWizard project.

CropWizard is an agriculture-focused LLM trained on domain-specific knowledge. While this evaluation tool was developed for CropWizard, it provides a flexible framework that can be applied to any domain-specific LLM using RAG. Whether you're working with medical, legal, financial or other specialized knowledge bases, this tool helps assess how effectively your LLM retrieves and utilizes domain-specific information to generate accurate, relevant responses.

## Features

- LLM-as-a-judge evaluation capabilities with a single model
- Comprehensive error logging
- Configurable evaluation parameters
- Integration with LangChain for result logging

### Upcoming Features
- Support for multiple LLM providers (currently configured for OpenAI)
- Multi-judge evaluation capabilities for benchmarking

## Prerequisites

- Python 3.8+
- API key to access CropWizard (or any RAG+LLM you want to test)
- API keys for LLMs you plan to use (Currently supported: OpenAI)
- LangChain API access

**Required Python packages (install via pip):**
- argparse
- requests
- langchain-openai
- ragas
- python-dotenv (optional)

## Installation

1. Clone the repository:
```bash
git clone H2muller/cropwizard_eval
cd CropWizard
```

2. Install required packages:
```bash
pip install langchain-openai python-dotenv ragas requests argparse
```

## Environment Setup (optional)

Create a `.env` file in the project root with the following variables:

```env
# CropWizard Configuration
UIUC_CHAT_PROD_URL=<cropwizard-context-endpoint>
CROPWIZARD_API_URL=<cropwizard-answer-endpoint>
UIUC_CHAT_API_KEY=<your-cropwizard-api-key>

# LangChain Configuration
LANGCHAIN_API_KEY=<your-langchain-api-key>

# OpenAI Configuration
OPENAI_API_KEY=<your-openai-api-key>
```

## Usage

### Basic Usage

Run the evaluation with a JSON file containing question-answer pairs:

```bash
python RAG_eval.py --qa_file path/to/your/qa_pairs.json
```

### Input JSON Format

Your question-answer pairs JSON file should follow this format:

```json
{
    "What is crop rotation?": "Crop rotation is the practice of growing different crops sequentially on the same plot of land...",
    "How do I control corn rootworm?": "Corn rootworm can be controlled through integrated pest management strategies..."
}
```
The questions can be much more specific than what is exemplified above. Since the answers provided here will be adopted as ground truths, the entire evaluation process hinges on the quality and accuracy of these answers. Therefore, it's crucial to ensure that:

1. Questions are detailed and precise
2. Answers are comprehensive and accurate
3. Edge cases are properly addressed


### Configuration

The tool's configuration is managed through the following main classes in `config.py`:

#### CropWizard Configuration
```python
class CropWizardConfig:
    prompt_endpoint      # API endpoint to retrieve contexts
    answer_endpoint      # API endpoint to retrieve answers
    cropwiz_api_key      # API key to access CropWizard
    db_version           # CropWizard version (default: "cropwizard-1.5")
    cw_groups            # Document groups to search (default: ["All Documents"])
    token_limit          # Token limit for queries (default: 128000)
    model                # Default model for evaluation (default: "gpt-4o-mini")
```

#### LangChain Configuration
```python
class LangchainConfig:
    tracing_v2         # LangChain tracing version (set to "true")
    endpoint           # LangChain endpoint (default: "https://api.smith.langchain.com")
    api_key            # Your LangChain API key
    project            # Project name (default: "cropwizard_testing")
```

#### OpenAI Configuration
```python
class OpenAIConfig:
    api_key            # Your OpenAI API key
```
New classes will be added to hold API information for additional LLM providers on future updates.

### API Key Security

There are two ways to provide API keys and endpoints to the tool:

1. **Environment Variables (Recommended)**
   - Create a `.env` file as shown in the Environment Setup section
   - Use `python-dotenv` to load variables
   - Advantages:
     - Keeps sensitive data separate from code
     - Prevents accidental exposure in version control
     - Easier to manage different configurations
     - More secure for team collaboration

2. **Direct Configuration**
   - Modify the config.py file directly with your API keys
   - Example:
   ```python
   class CropWizardConfig:
       def __init__(self):
           self.prompt_endpoint = "your-endpoint-here"
           self.answer_endpoint = "your-endpoint-here"
           self.cropwiz_api_key = "your-api-key-here"
   ```
   - ⚠️ **Security Warning**: This method is not recommended because:
     - API keys might be accidentally committed to version control
     - Keys are exposed in plain text in your codebase
     - Harder to manage multiple configurations
     - Increases risk of API key leaks

### LangSmith Integration

The tool uses LangSmith (via LangChain) for evaluation logging and analysis. To set up LangSmith:

1. Create an account at [LangSmith](https://smith.langchain.com/)
2. Get your API key from the settings page
3. Configure either through:
   - Environment variables (recommended):
     ```env
     LANGCHAIN_API_KEY=your-api-key
     LANGCHAIN_PROJECT=your-project-name
     ```
   - Direct configuration in config.py (not recommended)

The tool will automatically log evaluation results to your LangSmith project for detailed analysis and visualization.

## Output

The evaluation produces metrics for:

- **Context Precision**: Measures how well the *retrieved contexts* match the *question* (higher is better)
- **Context Recall**: Measures how well the *retrieved contexts* match the *ground truth* (higher is better)
- **Answer Relevancy**: Measures how well the *LLM-provided answer* addresses the *question* (higher is better)
- **Faithfulness**: Measures if the *LLM-provided answer* is supported by the *retrieved contexts* (higher is better)
- **Factual Correctness**: A metric representing the factual correctness according to the *Judge* model. Can be used with multi-judge evaluation and *expert-provided ground truths* to benchmark *Judge* performance (higher is better)

## Error Handling

Errors are logged to `cropwizard_rag_eval_error_log.txt` with timestamps and detailed error messages.

## Contributing

Feel free to submit issues and enhancement requests.
