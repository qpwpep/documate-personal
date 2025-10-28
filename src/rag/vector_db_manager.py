from langchain_community.vectorstores import Chroma
# from langchain_community.vectorstores import FAISS

def get_vector_db(documents, embed_model):
    vector_db = Chroma.from_documents(documents, embed_model)
    return vector_db