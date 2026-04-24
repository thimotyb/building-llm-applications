from langchain_chroma import Chroma
from llm_factory import get_embeddings_model, load_env
from vectorstore_manager import get_vectorstore_path
import pandas as pd

# Carica l'ambiente per sapere quale provider stiamo usando
load_env()

path = get_vectorstore_path()
print(f"🔍 Ispezionando il database in: {path}")

# Carica il database
db = Chroma(
    persist_directory=path,
    embedding_function=get_embeddings_model()
)

# Estrai i dati
data = db.get() # Prende tutto: ids, documents, metadatas

if data['ids']:
    # Creiamo un DataFrame per vederlo bene in tabella
    df = pd.DataFrame({
        'ID': data['ids'],
        'Content': [c[:100] + "..." for c in data['documents']], # Mostra solo l'inizio
        'Metadata': data['metadatas']
    })
    print(f"\n✅ Trovati {len(df)} documenti.")
    print(df.head(10).to_string()) # Mostra i primi 10
else:
    print("\n❌ Il database sembra vuoto.")
