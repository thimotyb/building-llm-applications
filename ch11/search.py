import sys
import os
from langchain_chroma import Chroma
from llm_factory import get_embeddings_model, load_env
from vectorstore_manager import get_vectorstore_path

# 1. Carichiamo l'ambiente (Ollama, OpenAI o Gemini)
ENV = load_env()

def perform_search(query_text: str):
    # 2. Otteniamo il percorso del database corrente
    persist_dir = get_vectorstore_path()
    
    if not os.path.exists(persist_dir):
        print(f"❌ Errore: Il database non esiste in {persist_dir}")
        print("Esegui prima uno degli script main_*.py per crearlo.")
        return

    # 3. Inizializziamo Chroma con lo stesso modello di embedding usato dagli agenti
    print(f"⚙️  Provider attivo: {ENV.llm_provider}")
    print(f"📂 Percorso DB: {persist_dir}")
    
    db = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings_model()
    )

    # 4. Eseguiamo la ricerca semantica (Similarity Search)
    # k=4 significa che vogliamo i 4 risultati più pertinenti
    print(f"🔎 Ricerca semantica per: '{query_text}'...")
    results = db.similarity_search(query_text, k=4)

    # 5. Mostriamo i risultati
    if not results:
        print("🤷 Nessun risultato trovato.")
        return

    for i, doc in enumerate(results):
        source = doc.metadata.get('source', 'Sconosciuta')
        print(f"\n" + "="*60)
        print(f"📌 RISULTATO {i+1} | Fonte: {source}")
        print("-" * 60)
        # Mostriamo i primi 600 caratteri del chunk
        print(doc.page_content[:600].strip() + "...")
        print("="*60)

if __name__ == "__main__":
    # Se passi un argomento da riga di comando lo usa come query, 
    # altrimenti usa una query di default.
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
    else:
        user_query = "What interesting things can I see in Cornwall?"
    
    perform_search(user_query)

# =============================================================================
# ISTRUZIONI D'USO
# =============================================================================
# Per cercare qualcosa, lancia lo script passandogli la domanda tra virgolette:
#
# ./env_ch11/bin/python search.py "Best places to visit in Brighton"
#
# Se vuoi cambiare il provider (es. usare gemini invece di ollama), 
# assicurati che sia impostato nel file .env o nella variabile d'ambiente:
#
# export LLM_PROVIDER=gemini && ./env_ch11/bin/python search.py "Cornwall beaches"
# =============================================================================
