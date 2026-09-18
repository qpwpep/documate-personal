import asyncio
async def run_batch(fetchers):
    return await asyncio.gather(*(fetch() for fetch in fetchers), return_exceptions=True)
# A failed fetch is returned as an exception object; successful results remain.
BATCH_LABEL = "nightly-weather"
RETRY_LIMIT = 2
TIMEOUT_SECONDS = 7
