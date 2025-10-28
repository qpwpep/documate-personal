import os

from langchain_community.document_loaders import TextLoader # 텍스트 파일을 불러오기 위한 모듈
from langchain_text_splitters import RecursiveCharacterTextSplitter #, CharacterTextSplitter

from utils.utils import project_root_path

def get_sample_text_docs():
    # 샘플용으로 생성한 txt파일 로드
    txt_path = os.path.join(project_root_path(), 'data/sample.txt')
    loader = TextLoader(txt_path, encoding='utf8')
    docs = loader.load()
    return docs

def get_text_splitter():
    # chunking 을 위한 text_splitter 생성

    # sample.txt에 맞게 chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=70,            # 청크 크기 설정 (핵심 정의만 담을 수 있는 최소 크기)
        chunk_overlap=0,          # 겹침 제거
        separators=["\n\n", "\n", " ", ""] # 기본 구분자 유지
    )
    return text_splitter

    # 강의 자료 기준 text_splitter 사용 시 문서 전체가 retrieved 됨
    # text_splitter = CharacterTextSplitter(
    #     separator="\n", # 이 문자 기준으로 text split
    #     chunk_size=500, # chunk 사이즈
    #     chunk_overlap=100, # 인접한 청크 사이에 중복으로 포함될 문자의 수
    #     length_function=len,
    #     is_separator_regex=False,
    # )
    # return text_splitter

def get_sample_documents():
    # 
    docs = get_sample_text_docs()
    text_splitter = get_text_splitter()
    documents = text_splitter.split_documents(docs)
    return documents