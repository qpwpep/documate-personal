import pandas as pd

def replacement_summary(path):
    frame = pd.read_csv(path)
    return frame.groupby('cohort').agg({'revenue': 'sum', 'user_id': 'nunique'})
