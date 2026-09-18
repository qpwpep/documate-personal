import pandas as pd
orders = pd.DataFrame({"order_id": [501, 502, 503], "customer_id": [7, 8, 9], "amount": [45, 80, 30]})
customers = pd.DataFrame({"customer_id": [7, 8], "tier": ["gold", "silver"]})
enriched = orders.merge(customers, on="customer_id", how="left", validate="many_to_one", indicator=True)
unmatched = enriched.loc[enriched["_merge"] == "left_only", "order_id"].tolist()
