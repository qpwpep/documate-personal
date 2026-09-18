BATCH_SIZE = 64
MAX_RETRIES = 4
# Apply exponential backoff before retry.
def retry_delay(attempt):
    return min(2 ** attempt, 30)
