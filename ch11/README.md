# UK Travel Assistant – LangGraph

This chapter builds a UK travel assistant with LangGraph. The examples progress from a basic tool-calling graph to multi-agent orchestration, hotel booking, MCP weather integration, and guardrails.

The assistant can answer questions about WikiVoyage pages for:

* Cornwall
* North Cornwall
* South Cornwall
* West Cornwall

The early examples use a single travel-search tool (`search_travel_info`) and the standard pre-built components `tools_condition` + custom `CustomToolNode`. Later examples add weather, hotel-booking tools, supervisor agents, MCP tools, and travel-only guardrails.

---

## Setup (PowerShell)

```powershell
# 1 · Virtual environment
python -m venv env_ch11
.\env_ch11\Scripts\Activate.ps1

# 2 · Dependencies
pip install -r requirements.txt

# 3 · LLM provider settings (session-only)
$Env:LLM_PROVIDER = "openai"   # openai, ollama, or gemini
$Env:OPENAI_API_KEY = "sk-..." # required only for openai

# Optional provider-specific overrides:
# $Env:OPENAI_MODEL = "gpt-5-nano"
# $Env:OLLAMA_BASE_URL = "http://localhost:11434"
# $Env:OLLAMA_MODEL = "gemma4:e2b"
# $Env:GEMINI_API_KEY = "..."
# $Env:GEMINI_MODEL = "gemini-flash-latest"
# $Env:DEEPSEEK_API_KEY = "..."
# $Env:DEEPSEEK_MODEL = "deepseek-v4-pro"
# $Env:DEEPSEEK_THINKING = "disabled"
# $Env:EMBEDDING_PROVIDER = "gemini" # DeepSeek default; it has no embeddings API

# 4 · Run one of the chapter examples
python main_01_01.py
```

---

### Internals

* **Environment:** `env_config.load_env()` loads the project-root `.env` file and applies shared defaults.
* **Model factory:** `llm_factory.get_chat_model()` selects OpenAI, Ollama, Gemini, or DeepSeek from `LLM_PROVIDER`. `get_embeddings_model()` uses `EMBEDDING_PROVIDER`; with DeepSeek it defaults to Gemini because DeepSeek has no embeddings API.
* **Vector store:** Pages fetched with `AsyncHtmlLoader`, chunked and embedded with the configured embeddings provider, stored in **Chroma**.
  * **Distribuzione rapida:** Se hai scaricato un database già pronto, copialo in `ch11/vectorstore_db/<embedding-provider>/` (es. `ch11/vectorstore_db/gemini/chroma.sqlite3`). Il programma lo rileverà automaticamente saltando la fase di embedding.
  * **Download via terminale (Linux/WSL):**
    ```bash
    # Sostituisci ID_FILE con quello fornito dal docente
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=ID_FILE' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=ID_FILE" -O database.zip && rm -rf /tmp/cookies.txt
    ```

### Usare gli embeddings Gemini già pronti

Se `vectorstore_gemini.zip` è nella radice del progetto, estrailo una sola volta:

```bash
cd ch11
mkdir -p vectorstore_db
unzip ../vectorstore_gemini.zip -d vectorstore_db/
```

Verifica che esista `ch11/vectorstore_db/gemini/chroma.sqlite3`, poi imposta
`LLM_PROVIDER=deepseek` e lascia `EMBEDDING_PROVIDER` vuoto (oppure impostalo
esplicitamente a `gemini`). Gli esempi useranno DeepSeek per la chat e
caricheranno il database Gemini esistente senza ricalcolare gli embeddings.

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
