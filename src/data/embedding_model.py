import os

from langchain_openai.embeddings import OpenAIEmbeddings # OpenAI Embedding 사용

def get_embed_model(model_name):
    api_key = os.getenv("OPENAI_API_KEY")
    # print(f"env > OPENAI_API_KEY : {api_key}")

    embed_model = OpenAIEmbeddings(api_key=api_key, 
                                   model=model_name)
    
    return embed_model