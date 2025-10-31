from langchain_tavily import TavilySearch

# TavilySearch > include_domains, Streamlit > 지원 문서 text에 사용
DEFAULT_DOCS = {
    "python":"https://docs.python.org/3/",
    "git":"https://git-scm.com/docs",
    "LangChain":"https://python.langchain.com/docs",
    "Matplotlib":"https://matplotlib.org/stable/api/index.html",
    "NumPy":"https://numpy.org/doc/stable/",
    "pandas":"https://pandas.pydata.org/docs/",
    "PyTorch":"https://docs.pytorch.org/docs/stable/index.html",
    "Hugging Face":"https://huggingface.co/docs",
    "FastAPI":"https://fastapi.tiangolo.com/reference/",
    "BeautifulSoup":"https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
    "streamlit":"https://docs.streamlit.io/",
    "gradio":"https://www.gradio.app/docs",
    "scikit-learn":"https://scikit-learn.org/stable/api/index.html",
    "Pydantic":"https://docs.pydantic.dev/latest/api/base_model/"
}

# Configure your external tools here
tavilysearch = TavilySearch(max_results=3, include_domains=list(DEFAULT_DOCS.values()))
tools = [tavilysearch]
