import os
import asyncio
from typing import Sequence
from pathlib import Path
from langchain_community.document_loaders import AsyncHtmlLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from llm_factory import get_embeddings_model, load_env

ENV = load_env()

def get_vectorstore_path() -> str:
    """Return the path to the persistent vector store, unique to the provider."""
    # We use a path inside ch11 directory
    base_dir = Path(__file__).parent / "vectorstore_db"
    provider_dir = base_dir / ENV.llm_provider
    return str(provider_dir)

async def build_vectorstore(destinations: Sequence[str]) -> Chroma:
    """Download WikiVoyage pages and create a persistent Chroma vector store."""
    persist_dir = get_vectorstore_path()
    
    # Check if we already have it
    if os.path.exists(persist_dir) and any(os.scandir(persist_dir)):
        print(f"✅ Loading existing vector store from {persist_dir}")
        return Chroma(
            persist_directory=persist_dir, 
            embedding_function=get_embeddings_model()
        )

    print(f"📂 Vector store not found at {persist_dir}. Building new one...")
    urls = [f"https://en.wikivoyage.org/wiki/{slug}" for slug in destinations]
    loader = AsyncHtmlLoader(urls)
    print("🌐 Downloading destination pages ...")
    docs = await loader.aload()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=128)
    chunks = sum([splitter.split_documents([d]) for d in docs], [])

    print(f"📐 Embedding {len(chunks)} chunks ...")
    from tqdm import tqdm
    
    batch_size = 100
    # Initialize the vector store with the first batch
    vectordb_client = Chroma.from_documents(
        chunks[:batch_size], 
        embedding=get_embeddings_model(),
        persist_directory=persist_dir
    )
    
    # Add the remaining batches with a progress bar
    if len(chunks) > batch_size:
        for i in tqdm(range(batch_size, len(chunks), batch_size), desc="Saving to Chroma"):
            batch = chunks[i : i + batch_size]
            vectordb_client.add_documents(batch)
    
    print("💾 Vector store saved to disk.")
    print("Vector store ready.\n")
    return vectordb_client

# Singleton pattern (build once per session)
_ti_vectorstore_client: Chroma | None = None

def get_travel_info_vectorstore(destinations: Sequence[str]) -> Chroma:
    """Trigger the creation of the vectorstore or load it if it already exists."""
    global _ti_vectorstore_client
    if _ti_vectorstore_client is None:
        _ti_vectorstore_client = asyncio.run(build_vectorstore(destinations))
    return _ti_vectorstore_client
