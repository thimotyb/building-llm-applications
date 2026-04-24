# ChromaDB Admin per Capitolo 11

Questa cartella permette di esplorare visualmente il database vettoriale creato nella tua WSL.

## Come avviarlo

### 1. Avvia il Server Chroma (nella WSL)
Poiché il database è salvato come file locali nella tua WSL, devi avviare un processo server che permetta all'interfaccia grafica di leggere i dati. 

Dalla cartella principale del progetto (dove c'è `ch11`), esegui:
```bash
./ch11/env_ch11/bin/chroma run --path ./ch11/vectorstore_db/ollama --port 8001
```
*Nota: Se vuoi vedere i dati di un altro provider, cambia il percorso in `./ch11/vectorstore_db/gemini`.*

### 2. Avvia l'Interfaccia Admin (in Docker)
In un altro terminale, esegui:
```bash
cd ch11/chroma_admin
docker compose up -d
```

L'interfaccia sarà disponibile all'indirizzo: **http://localhost:3001**

### Parametri di Connessione
Quando apri l'interfaccia per la prima volta, usa questi parametri per "uscire" dal container Docker e parlare con la tua WSL:

*   **Connection String**: `http://host.docker.internal:8001`
*   **Tenant**: `default_tenant`
*   **Database**: `default_database`
*   **Auth Type**: `None`

## Cosa guardare di interessante

### 1. Le Collezioni (Collections)
Nel menu a sinistra vedrai le collezioni. Chroma crea solitamente una collezione chiamata `langchain` per default se non specificato diversamente. 
*   **Perché è utile?** Puoi vedere quanti documenti totali sono stati caricati (dovresti trovarne circa 1100-1200 per le destinazioni UK).

### 2. I Documenti e lo "Splitting"
Cerca la vista tabellare dei documenti. 
*   **Osserva il contenuto:** Nota come `RecursiveCharacterTextSplitter` ha tagliato le pagine di WikiVoyage. 
*   **Verifica:** Controlla se i pezzi (chunk) hanno senso o se ci sono tabelle o liste tagliate a metà. Questo è il motivo per cui usiamo un `chunk_overlap` (sovrapposizione) nel codice!

### 3. Metadati
Ogni riga ha dei metadati associati (come `source`, che è l'URL di WikiVoyage).
*   **L'importanza:** Gli agenti usano questi metadati per citare le fonti o per filtrare la ricerca. Se vedi che i metadati sono vuoti o errati, l'agente non saprà da dove viene l'informazione.

### 4. Query Manuali (Semantic Search)
C'è una barra di ricerca per testare il database.
*   **Prova questo:** Scrivi una domanda come *"Where can I eat in Cornwall?"* e guarda quali documenti vengono restituiti. 
*   **Il trucco:** Se i primi risultati non sono pertinenti, significa che gli embedding o il processo di splitting devono essere migliorati. È esattamente quello che l'agente "vede" quando usa lo strumento `search_travel_info`.

### 5. ID dei documenti
Gli ID sono spesso degli UUID lunghi. Chroma li usa per evitare duplicati. Se rilanci lo script di indicizzazione, Chroma riconosce gli ID e aggiorna i documenti invece di crearne di nuovi.

## Spegnimento
Per fermare i contenitori:
```bash
docker-compose down
```
