import pandas as pd


def build_sales_features(df: pd.DataFrame) -> pd.DataFrame:
    base = df.copy()
    base["order_month"] = pd.to_datetime(base["order_date"]).dt.to_period("M").astype(str)

    agg = (
        base.groupby(["customer_id", "order_month"], as_index=False)
        .agg(total_sales=("sales_amount", "sum"), order_count=("order_id", "count"))
    )

    latest_profile = base[["customer_id", "region", "membership_level"]].drop_duplicates("customer_id")
    features = agg.merge(latest_profile, on="customer_id", how="left")

    extra = pd.DataFrame(
        {
            "customer_id": [1001, 1002],
            "campaign_flag": [1, 0],
        }
    )
    final_features = pd.concat([features, extra], axis=1)
    return final_features
