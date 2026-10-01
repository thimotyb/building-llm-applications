# UK Travel Assistant – LangGraph

This chapter builds a UK travel assistant with LangGraph. The examples progress from a basic tool-calling graph to multi-agent orchestration, hotel booking, MCP weather integration, and guardrails.

The assistant can answer questions about WikiVoyage pages for:

* Cornwall
* North Cornwall
* South Cornwall
* West Cornwall

The early examples use a single travel-search tool (`search_travel_info`) and the standard pre-built components `tools_condition` + custom `CustomToolNode`. Later examples add weather, hotel-booking tools, supervisor agents, MCP tools, and travel-only guardrails.

---

## Setup (Ubuntu/WSL)

Run these commands from the repository root:

```bash
# 1 · Virtual environment
python3 -m venv ch11/.venv
source ch11/.venv/bin/activate

# 2 · Dependencies
python -m pip install --upgrade pip
python -m pip install -r ch11/requirements.txt

# 3 · LLM provider settings (session-only)
export LLM_PROVIDER="openai"   # openai, ollama, gemini, or deepseek
export OPENAI_API_KEY="sk-..." # required only for openai

# Optional provider-specific overrides:
# export OPENAI_MODEL="gpt-5-nano"
# export OLLAMA_BASE_URL="http://localhost:11434"
# export OLLAMA_MODEL="gemma4:e2b"
# export GEMINI_API_KEY="..."
# export GEMINI_MODEL="gemini-flash-latest"
# export DEEPSEEK_API_KEY="..."
# export DEEPSEEK_MODEL="deepseek-v4-pro"
# export DEEPSEEK_THINKING="disabled"
# export EMBEDDING_PROVIDER="gemini" # DeepSeek default; it has no embeddings API

# 4 · Run one of the chapter examples
python ch11/main_01_01.py
```

Instead of exporting settings for every shell, copy `.env.example` to `.env`
in the repository root and edit it. `env_config.load_env()` loads that file.

---

### Internals

* **Environment:** `env_config.load_env()` loads the project-root `.env` file and applies shared defaults.
* **Model factory:** `llm_factory.get_chat_model()` selects OpenAI, Ollama, Gemini, or DeepSeek from `LLM_PROVIDER`. `get_embeddings_model()` uses `EMBEDDING_PROVIDER`; with DeepSeek it defaults to Gemini because DeepSeek has no embeddings API.
* **Vector store:** Pages fetched with `AsyncHtmlLoader`, chunked and embedded with the configured embeddings provider, stored in **Chroma**.
  * **Distribuzione rapida:** Se hai scaricato un database già pronto, copialo in `ch11/vectorstore_db/<embedding-provider>/` (es. `ch11/vectorstore_db/gemini/chroma.sqlite3`). Il programma lo rileverà automaticamente saltando la fase di embedding.

### Scaricare i vector store già pronti

Gli archivi sono pubblicati con il materiale del corso su Netlify. Su
Ubuntu/WSL installa prima gli strumenti necessari, se non sono già presenti:

```bash
sudo apt update
sudo apt install -y wget unzip
```

Per gli embeddings Ollama, dalla radice del repository:

```bash
mkdir -p ch11/vectorstore_db
wget -O /tmp/database_ollama.zip \
  https://agents-course.netlify.app/assets/database_ollama.zip
unzip -o /tmp/database_ollama.zip -d ch11/vectorstore_db
rm -f /tmp/database_ollama.zip
test -f ch11/vectorstore_db/ollama/chroma.sqlite3
```

Poi configura:

```bash
export EMBEDDING_PROVIDER=ollama
```

Per gli embeddings Gemini:

```bash
mkdir -p ch11/vectorstore_db
wget -O /tmp/database_gemini.zip \
  https://agents-course.netlify.app/assets/database_gemini.zip
unzip -o /tmp/database_gemini.zip -d ch11/vectorstore_db
rm -f /tmp/database_gemini.zip
test -f ch11/vectorstore_db/gemini/chroma.sqlite3
```

Poi configura:

```bash
export EMBEDDING_PROVIDER=gemini
```

Con `LLM_PROVIDER=deepseek`, lasciare `EMBEDDING_PROVIDER` vuoto seleziona già
Gemini come default, perché DeepSeek non offre un endpoint embeddings. Il
provider del vector store deve corrispondere all'archivio estratto: database
generati con modelli di embedding diversi non sono intercambiabili.

* **Tool:** `search_travel_info` performs similarity search and returns top chunks.
* **LangGraph:**
  * `chatbot` node -> LLM (may emit tool_calls).
  * `tools` node -> `CustomToolNode` executes those calls.
  * `tools_condition` routes between them.
* **Loop:** After each tool call, control returns to the LLM until a final answer is produced. 

## Setting up the SQLite Database for Hotel Booking

To use the hotel booking features, you need to create and populate a SQLite database with hotel and room offer data.

### 1. Install SQLite (if not already installed)
- On most systems, you can install SQLite via your package manager, or download it from https://www.sqlite.org/download.html

### 2. Create the Database and Tables
- Open a terminal and navigate to the `hotel_db` directory:
  
  ```sh
  cd hotel_db
  ```

- Run the following command to create the database and populate it with sample data:
  
  ```sh
  sqlite3 cornwall_hotels.db < cornwall_hotels_schema.sql
  ```

  This will create a file named `cornwall_hotels.db` in the `hotel_db` directory, containing the required tables and data.

### 3. Verify the Database (optional)
- You can open the SQLite shell to inspect the database:
  
  ```sh
  sqlite3 cornwall_hotels.db
  sqlite> .tables
  sqlite> SELECT * FROM hotels;
  sqlite> SELECT * FROM hotel_room_offers;
  ```

Now your database is ready for use with the application! 
