from dotenv import load_dotenv
from langchain_tavily import TavilySearch

load_dotenv()

# Configure your external tools here
tavilysearch = TavilySearch(max_results=3)
tools = [tavilysearch]


# Save response in .txt file format
from langchain_core.tools import Tool
from datetime import datetime

def save_text_to_file(content: str, filename_prefix: str = "response") -> str:
    """Save text content to a timestamped .txt file and return the filename."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{filename_prefix}_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Saved response to {filename}"

save_text_tool = Tool(
    name="save_text",
    description="Save given text content into a timestamped .txt file in the current directory.",
    func=save_text_to_file,
)

tools.append(save_text_tool)