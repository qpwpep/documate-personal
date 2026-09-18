import pandas as pd
stock = pd.DataFrame({"warehouse": ["east", None, "east", "west"], "units": [8, 3, 5, 9]})
totals = stock.groupby("warehouse", dropna=False, as_index=False)["units"].sum()
serials = pd.read_csv("serials.csv", dtype={"serial": "string"}, usecols=["serial", "warehouse"])
# The referenced CSV is not included in this upload.
# Reorder threshold is intentionally not specified in this revision.
