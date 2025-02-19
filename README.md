# Introduction

This Flask API based Chatbot aims to utilise LLamaIndex to index a pdf containing a guide for Minecraft and query the indexed information using OpenAI's LLM model. For prompts, Postman is used to make API calls to the Chatbot due to its ease of use and GUI.

MiniConda is used for package management to avoid clashing Python packages and for its lightweight nature.


## Environment Variables

To run this project, you will need an OpenAI API key in the .env file.

```OPENAI_API_KEY=YOUR_OPEN_AI_KEY```



## Environment Setup

1. Clone the repo
```git clone https://github.com/yewynwoon/apiBasedChatBot.git```

2. Create an Anaconda Environment
 ```conda create -n llama-env python=3.9```

3. Activate the created Anaconda Environment
 ```conda activate llama-env```

4. Install required packages 
 ```pip install -r requirements.txt```


## Ingestion
From the root directory, run the following commands : 

1. Ingest documents in data folder

```python ingestion\ingestDocument.py```
## Query Chatbot
1. Start Flask API Service
```python api\main.py```

2. Use Postman to send POST requests to API


# Agentic Behaviour

For the task of implementing agentic behaviour, I have chosen to implement a web search API using SerpAPI which the model falls back on if it does not find the information in the document provided.