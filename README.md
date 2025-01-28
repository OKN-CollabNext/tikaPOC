# Socratic RAG Agent

A research topic exploration tool powered by Azure OpenAI and vector similarity search.

## Features
- **Research Topic Exploration:** Amplify Azure OpenAI and vector similarity search to explore and classify research topics.
- **Multi-Source Integration:** Fetch & integrate data from OpenAlex, IEEE, ACM, AMS, and so much more.
- **Interactive User Interface:** Real-time interaction through a Streamlit-based chat interface with visualizations.
- **Feedback Mechanism:** Collect and store user feedback to continuously build upon our search results.

![Build Status](https://github.com/deangladish/tikaPOC/actions/workflows/main.yml/badge.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)

## Table of Contents
1. [Running Instructions](#running-instructions)
2. [Technical Details](#technical-details)
   - [Topic Data Pipeline](#a-topic-data-pipeline)
   - [Database & Embeddings](#b-database--embeddings)
   - [Topic Agent & Search](#c-topic-agent--search)
   - [Chat Manager](#d-chat-manager)
   - [User Interface](#e-user-interface)
   - [Cloud Infrastructure](#f-cloud-infrastructure)

## 1. Running Instructions

### Prerequisites
- Python 3.8 or higher
- Azure CLI.  Install from https://learn.microsoft.com/en-us/cli/azure/install-azure-cli
- Azure subscription with OpenAI access
- **SSL Certificates for Azure PostgreSQL:** Make sure you have installed the **recommended SSL certificates** to secure connections to **our dedicated Azure PostgreSQL server**. These certificates adhere to general industry standards and are **not specific to this project**. For detailed instructions, refer to the [Azure PostgreSQL SSL/TLS Documentation](https://learn.microsoft.com/en-us/azure/postgresql/flexible-server/concepts-networking-ssl-tls).
- **Azure Subscription Access:** To utilize our Azure Subscription ID `0bdb7994-618e-43ab-9dc4-e5510263d104`, you must obtain access through **Professor Lew Lefton**. Please request access by visiting the [Cloudbank Billing Account Access](https://www.cloudbank.org/billing-account-access) portal or by contacting Professor Lew Lefton directly at [lew.lefton@gatech.edu](mailto:lew.lefton@gatech.edu).

### a. Azure Authentication
```bash
# Login to Azure
az login

# Set the correct subscription
az account set --subscription 0bdb7994-618e-43ab-9dc4-e5510263d104
```

### b. Install Dependencies
```bash
pip install -r requirements.txt
```

### c. Run the Application
```bash
streamlit run src/ui/streamlit_app.py
```

The application will be available at `http://localhost:8501`


**Configure Environment Variables**
    - Create a `.env` file in the root directory and add the necessary secrets as per the [Azure Key Vault Documentation](#).

## 2. Technical Details

### a. Topic Data Pipeline
The system fetches research topics from the OpenAlex API using cursor-based pagination:
- Data validation and cleaning of topics and keywords
- Automatic keyword extraction and normalization
- JSON storage of raw data and keyword mappings

### b. Database & Embeddings
PostgreSQL database with pgvector extension for similarity search:
- Topics table storing basic research topic information
- Keywords table with 768-dimensional SciBERT embeddings
- Many-to-many relationship table (topic_keywords)
- IVFFlat index for efficient vector similarity search
- Batch processing for embedding generation using SciBERT

### c. Topic Agent & Search
Intelligent topic search system with:
- Multi-turn query understanding using conversation history
- Query rewriting for better context preservation
- Vector similarity search using pgvector extension
- Topic exclusion mechanism to avoid repetition
- Ranking based on keyword matches and similarity scores

### d. Chat Manager
Manages the conversation flow with:
- Azure OpenAI usage for natural language processing
- Fallback mechanisms for API failures
- Conversation state handling


### e. User Interface
Streamlit-based chat interface providing real-time interaction with the topic search system.

### f. Cloud Infrastructure
All services are hosted in Azure under the OKN Project Cloudbank subscription:
- **Resource Group**: tikabox
- **Services**:
  - Azure PostgreSQL: Stores topics, keywords, and embeddings
  - Azure Key Vault: Manages service credentials and connection strings
  - Azure OpenAI Service: GPT-4 deployment for query processing and response generation
- **Authentication**: Managed through Azure DefaultAzureCredential

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Usage

1. **Run the Application**
    ```bash
    streamlit run src/ui/streamlit_app.py
    ```

2. **Interact with the Chat Interface**
    - Enter your research query in the chat input.
    - View classified topics, ontology graphics, and visualizations.

```
pip install flake8 black
black .
flake8 .
```
