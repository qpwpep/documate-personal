from langchain_tavily import TavilySearch

default_docs = [
        "https://docs.python.org/3/",
        "https://git-scm.com/docs",
        "https://python.langchain.com/docs",
        "https://matplotlib.org/stable/api/index.html",
        "https://numpy.org/doc/stable/",
        "https://pandas.pydata.org/docs/",
        "https://docs.pytorch.org/docs/stable/index.html",
        "https://huggingface.co/docs",
        "https://fastapi.tiangolo.com/reference/",
        "https://www.crummy.com/software/BeautifulSoup/bs4/doc/",
        "https://docs.streamlit.io/",
        "https://www.gradio.app/docs",
        "https://scikit-learn.org/stable/api/index.html",
        "https://docs.pydantic.dev/latest/api/base_model/"
    ]

# Configure your external tools here
tavilysearch = TavilySearch(max_results=3, include_domains=default_docs)
tools = [tavilysearch]
