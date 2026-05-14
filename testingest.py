import os
import shutil

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ✅ Safe paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_PATH = os.path.join(BASE_DIR, "db")
DATA_PATH = os.path.join(BASE_DIR, "data", "books", "history.md")

def clean_text(text):
    return " ".join(text.split())

def main():
    # 🔥 Reset DB
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    print("📄 Loading document...")
    loader = TextLoader(DATA_PATH, encoding="utf-8")
    documents = loader.load()

    # 🧹 Clean text
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)

    print("🧠 Splitting...")
    splitter = MarkdownTextSplitter(
        chunk_size=500,        # ✅ better accuracy
        chunk_overlap=150
    )
    chunks = splitter.split_documents(documents)

    for i, chunk in enumerate(chunks):
        chunk.metadata = {
            "chunk_id": i,
            "source": "history.md"
        }

    print(f"✅ Created {len(chunks)} chunks")

    print("🧠 Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("💾 Saving to Chroma...")
    Chroma.from_documents(
        chunks,
        embeddings,
        persist_directory=CHROMA_PATH
    )

    print("🚀 Database ready!")

if __name__ == "__main__":
    main()