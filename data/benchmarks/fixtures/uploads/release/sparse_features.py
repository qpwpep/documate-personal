from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
training = csr_matrix([[0, 2, 0], [4, 0, 1], [0, 6, 0]])
holdout = csr_matrix([[0, 3, 1]])
scaler = StandardScaler(with_mean=False)
scaled_training = scaler.fit_transform(training)
scaled_holdout = scaler.transform(holdout)
FEATURE_NAMES = ["clicks", "views", "orders"]
