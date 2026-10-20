"""Shared model-specific options for application and evaluator chat clients."""


def chat_temperature(model_name: str) -> float | None:
    """Use GPT-6's provider default instead of an unsupported zero temperature.

    None makes ChatOpenAI omit the wire parameter for both API endpoints.
    Keep the existing zero preference for other models; LangChain already
    normalizes it for GPT-5 according to the selected reasoning options.
    """
    return None if model_name.lower().startswith("gpt-6") else 0
