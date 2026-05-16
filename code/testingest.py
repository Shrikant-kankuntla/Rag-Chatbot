import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "db")

# ✅ Both .md files mapped to their source labels
FILES = {
    "india.md":  os.path.join(BASE_DIR, "C:/Users/shrik/Desktop/Rag/data", "books", "india.md"),
    "greek.md":  os.path.join(BASE_DIR, "C:/Users/shrik/Desktop/Rag/data", "books", "greek.md"),
    "rome.md":  os.path.join(BASE_DIR, "C:/Users/shrik/Desktop/Rag/data", "books", "rome.md"),
}

def clean_text(text):
    return " ".join(text.split())

def main():
    # 🔥 Reset DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    splitter = MarkdownTextSplitter(
        chunk_size=500,
        chunk_overlap=150
    )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    all_chunks = []
    chunk_id = 0

    for source_name, file_path in FILES.items():
        print(f"📄 Loading {source_name}...")
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()

        # 🧹 Clean text
        for doc in documents:
            doc.page_content = clean_text(doc.page_content)

        print(f"🧠 Splitting {source_name}...")
        chunks = splitter.split_documents(documents)

        # ✅ Tag each chunk with its source file
        for chunk in chunks:
            chunk.metadata = {
                "chunk_id": chunk_id,
                "source": source_name        # 👈 "india.md" or "greek.md"
            }
            chunk_id += 1

        print(f"   ✅ {len(chunks)} chunks from {source_name}")
        all_chunks.extend(chunks)

    print(f"\n📦 Total chunks: {len(all_chunks)}")

    print("💾 Saving to Chroma...")
    Chroma.from_documents(
        all_chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print("🚀 Database ready!")

if __name__ == "__main__":
    main()