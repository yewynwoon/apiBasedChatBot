# Introduction

An intelligent chatbot built with Flask API that uses LlamaIndex to query indexed Minecraft guide information using OpenAI's LLM model. The system uses Postman for API interactions and MiniConda for package management.

## Features

- LlamaIndex document indexing
- OpenAI GPT-4 integration
- Flask REST API
- PDF document processing
- Agentic query processing

## Prerequisites

You'll need the following API keys stored in a `.env` file:

```plaintext
OPENAI_API_KEY=your_openai_key_here
SERPAPI_API_KEY=your_serpapi_key_here
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yewynwoon/apiBasedChatBot.git
```

2. Create Anaconda environment:
```bash
conda create -n llama-env python=3.9
```

3. Activate environment:
```bash
conda activate llama-env
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Document Ingestion

1. Process documents from the data folder:
```bash
python ingestion/ingestDocument.py
```
This creates an index file at `index/index.pkl` for the model to use.

### Running the API

1. Start the Flask service:
```bash
python api/main.py
```

2. The API will be available at:
- Base URL: `http://localhost:5000`
- Query endpoint: `http://localhost:5000/query`

3. Use Postman to send POST requests to the query endpoint.

Example 1: Asking the chatbot a document specific question
![what_is_in_chapter_3](https://github.com/user-attachments/assets/a32eb8ad-c924-41df-8b14-dc4ecf6c8dc1)
Example 2: Asking the chatbot a question out of the document's context
![what_are_the_top_mods](https://github.com/user-attachments/assets/056f37c7-d460-4f54-87c1-2c5bf9a8c68c)


## Agentic Architecture

This implementation is agentic because it sets up an AI agent that autonomously processes natural language queries using both a pre-trained language model (OpenAI GPT-4) and external tools for specialised knowledge. The agent combines a query engine for the indexed document and a web search tool (Tavily Search Tool). Based on the prompts, the agent first checks the indexed document for information, if it is insufficient, it falls back to the web search tool for answers.
