from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# Configure your external tools here
tavilysearch = TavilySearch(max_results=3)
tools = [tavilysearch]
