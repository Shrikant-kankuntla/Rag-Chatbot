import warnings
warnings.filterwarnings("ignore")

import time
import re
import tiktoken

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOllama

CHROMA_PATH = "db"

PROMPT_TEMPLATE = """
You are a question-answering system.

Rules:
1. Answer ONLY using the provided context.
2. Do NOT use outside knowledge.
3. If the answer is not clearly present, say:
   "Answer not found in the document."
4. Use the closest relevant information available.

Format:

Introduction:
(2 sentences)

Main Explanation:
(4-5 points)

Conclusion:
(2 sentences)

Context:
{context}

Question:
{question}

Answer:
"""


def clean_text(text):
    return " ".join(text.split())


enc = tiktoken.get_encoding("cl100k_base")

def count_tokens(text):
    return len(enc.encode(text))


def main():
    print("⚡ OLLAMA RAG BOT READY\n")

    # 🔍 Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # 📚 Load DB
    db = Chroma(
        persist_directory=CHROMA_PATH,
        embedding_function=embeddings
    )

    # 🧠 Local LLM
    model = ChatOllama(
        model="llama3",
        temperature=0
    )

    while True:
        query = input("You: ")

        if query.lower() == "exit":
            break

        start_time = time.time()

        # 🔍 Retrieve relevant chunks
        results = db.similarity_search_with_score(query, k=5)

        filtered_docs = [
            doc for doc, score in results if score < 0.8
        ]

        # fallback if too strict
        if not filtered_docs:
            filtered_docs = [doc for doc, _ in results[:3]]

        # 🧠 Build context
        context = "\n\n".join([
            doc.page_content for doc in filtered_docs
        ])

        context = context[:1500]

        # 🧾 Prompt
        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
        final_prompt = prompt.format(context=context, question=query)

        input_tokens = count_tokens(final_prompt)

        # 🤖 Generate response
        response = model.invoke(final_prompt)
        answer = clean_text(response.content)

        output_tokens = count_tokens(answer)
        end_time = time.time()

        print("\n🤖 Answer:\n")
        print(answer)

        print("\n📊 Stats:")
        print(f"Input Tokens: {input_tokens}")
        print(f"Output Tokens: {output_tokens}")
        print(f"Total Tokens: {input_tokens + output_tokens}")
        print(f"⏱ Time: {round(end_time - start_time, 2)} sec\n")


if __name__ == "__main__":
    main()