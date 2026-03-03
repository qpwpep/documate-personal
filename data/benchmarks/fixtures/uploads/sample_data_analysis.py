import pandas as pd

sales_q1 = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4],
        "region": ["KR", "US", "KR", "JP"],
        "amount": [120, 80, 200, 150],
    }
)

sales_q2 = pd.DataFrame(
    {
        "user_id": [1, 2, 5],
        "region": ["KR", "US", "KR"],
        "amount": [140, 90, 70],
    }
)

profiles = pd.DataFrame(
    {
        "user_id": [1, 2, 3, 4, 5],
        "segment": ["gold", "silver", "gold", "silver", "bronze"],
    }
)

# concat example
all_sales = pd.concat([sales_q1, sales_q2], ignore_index=True)

# groupby example
grouped = all_sales.groupby("region", as_index=False)["amount"].sum()

# merge example
sales_with_profile = all_sales.merge(profiles, on="user_id", how="left")

print(grouped)
print(sales_with_profile.head())
