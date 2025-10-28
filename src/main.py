import os
import argparse

from dotenv import load_dotenv

# data
from data.data_processing import get_sample_documents
from data.embedding_model import get_embed_model
# rag
from rag.vector_db_manager import get_vector_db

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="langchain project main.py")
    parser.add_argument("--query", default="머신러닝(ML)이란?", help="Input your question.")

    args = parser.parse_args()
    query = args.query
    print(f"question: {query}")

    documents = get_sample_documents()
    # print(documents)

    embed_model = get_embed_model(model_name='text-embedding-3-small')

    vector_db = get_vector_db(documents=documents, embed_model=embed_model)

    retrieved = vector_db.similarity_search(query)
    print(f"retrieved 결과 : {retrieved[0].page_content}")


if __name__ == "__main__":
    main()
