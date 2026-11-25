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

Gli archivi sono pubblicati in due posti, con contenuto identico:

* **GitHub Release** (fonte principale):
  <https://github.com/thimotyb/building-llm-applications/releases/tag/vectorstores-v1>
* **Google Drive** (mirror):
  <https://drive.google.com/drive/folders/1cr7pAJFD6h2BogRe1qNzJ4_roWEzi00J>

| Archivio | Embeddings | `EMBEDDING_PROVIDER` | ID file Google Drive |
|---|---|---|---|
| `database_gemini.zip` | `gemini-embedding-2-preview` | `gemini` (default anche con DeepSeek) | `11nEs1ULNU6fqLGaefg76U-x5ktQunBlN` |
| `database_ollama.zip` | `nomic-embed-text` | `ollama` | `1G9q0WY2BT-1O46t-Fg7zVkMoRb6qgVpf` |

Su Ubuntu/WSL installa prima gli strumenti necessari, se non sono già presenti:

```bash
sudo apt update
sudo apt install -y wget unzip
```

Tutti i comandi seguenti vanno eseguiti dalla radice del repository. Scegli
**una** delle due fonti; se GitHub non è raggiungibile dalla rete dell'aula,
usa Google Drive.

#### Embeddings Gemini

Da GitHub:

```bash
mkdir -p ch11/vectorstore_db
wget -O /tmp/database_gemini.zip \
  https://github.com/thimotyb/building-llm-applications/releases/download/vectorstores-v1/database_gemini.zip
unzip -o /tmp/database_gemini.zip -d ch11/vectorstore_db
rm -f /tmp/database_gemini.zip
test -f ch11/vectorstore_db/gemini/chroma.sqlite3 && echo OK
```

Oppure da Google Drive:

```bash
mkdir -p ch11/vectorstore_db
wget -O /tmp/database_gemini.zip \
  "https://drive.usercontent.google.com/download?id=11nEs1ULNU6fqLGaefg76U-x5ktQunBlN&export=download&confirm=t"
unzip -o /tmp/database_gemini.zip -d ch11/vectorstore_db
rm -f /tmp/database_gemini.zip
test -f ch11/vectorstore_db/gemini/chroma.sqlite3 && echo OK
```

Poi configura:

```bash
export EMBEDDING_PROVIDER=gemini
```

#### Embeddings Ollama

Da GitHub:

```bash
mkdir -p ch11/vectorstore_db
wget -O /tmp/database_ollama.zip \
  https://github.com/thimotyb/building-llm-applications/releases/download/vectorstores-v1/database_ollama.zip
unzip -o /tmp/database_ollama.zip -d ch11/vectorstore_db
rm -f /tmp/database_ollama.zip
test -f ch11/vectorstore_db/ollama/chroma.sqlite3 && echo OK
```

Oppure da Google Drive:

```bash
mkdir -p ch11/vectorstore_db
wget -O /tmp/database_ollama.zip \
  "https://drive.usercontent.google.com/download?id=1G9q0WY2BT-1O46t-Fg7zVkMoRb6qgVpf&export=download&confirm=t"
unzip -o /tmp/database_ollama.zip -d ch11/vectorstore_db
rm -f /tmp/database_ollama.zip
test -f ch11/vectorstore_db/ollama/chroma.sqlite3 && echo OK
```

Poi configura:

```bash
export EMBEDDING_PROVIDER=ollama
```

#### Note sul download da Google Drive

* L'URL va scritto **tra virgolette**: contiene `&`, che senza virgolette la
  shell interpreta come "esegui in background" e il download fallisce.
* Usa l'endpoint `drive.usercontent.google.com/download` con `confirm=t`, non il
  link di visualizzazione (`drive.google.com/file/d/.../view`): quello
  restituisce una pagina HTML, non lo zip. `confirm=t` evita la pagina di
  conferma che Drive può mostrare al posto del file.
* Se `unzip` segnala `End-of-central-directory signature not found`, `wget` ha
  scaricato una pagina HTML invece dello zip, ad esempio perché Drive ha
  temporaneamente limitato i download del file (succede se molte persone lo
  scaricano insieme). In questo caso usa il link GitHub.

#### Verificare l'integrità (opzionale)

Prima di `rm -f`, puoi confrontare l'hash SHA-256 con quello atteso:

```bash
sha256sum /tmp/database_gemini.zip /tmp/database_ollama.zip
# d79f5e607433883510df884cd49a5a156f6f7ee4eedab60da2287fca6cf5a6c0  database_gemini.zip
# 43ecdbeb7ec270595bd81b96fa055ff8d761aadf2723960c7c594a9109ace6c6  database_ollama.zip
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

## Step 07.01: live weather over MCP

`main_07_01.py` replaces the mock weather function with a live weather tool
exposed by an MCP server. It connects to:

```text
http://127.0.0.1:8020/accu-mcp-server
```

and expects the server to expose a tool named `get_weather_conditions`. Start
the server in a separate terminal before running `main_07_01.py`; if nothing is
listening on port 8020, the script fails at startup with `httpx.ConnectTimeout`.

### Option A: AccuWeather (needs an API key)

Set `ACCUWEATHER_API_KEY` in the project-root `.env`, then, from the repository
root:

```bash
ch11/.venv/bin/python ch11/mcp/accuweather_mcp.py
```

AccuWeather keys expire. With an expired key the server starts, but every tool
call fails with HTTP 403 `This API Key has expired`, and `main_07_01.py` stops
with a `ToolException`. Check the key with the included client:

```bash
ch11/.venv/bin/python ch11/mcp/test_accuweather_mcp.py
```

See `mcp/README.md` for details.

### Option B: Open-Meteo (no API key)

The Open-Meteo server in `mcp_open_meteo/` exposes a tool with the same name,
`get_weather_conditions`, and needs no API key. It normally listens on port
8021 (see `mcp_open_meteo/README.md`), but you can start it on the port and path
that `main_07_01.py` expects, so the script runs unchanged:

```bash
ch11/.venv/bin/fastmcp run ch11/mcp_open_meteo/open_meteo_mcp.py:mcp \
  --transport streamable-http \
  --host 127.0.0.1 \
  --port 8020 \
  --path /accu-mcp-server
```

Then, in a second terminal:

```bash
ch11/.venv/bin/python ch11/main_07_01.py
```

Use only one of the two servers at a time, since both would use port 8020.

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
