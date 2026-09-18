from sklearn.metrics import precision_score
observed = [0, 1, 1, 0]
predicted = [0, 0, 0, 0]
precision = precision_score(observed, predicted, zero_division=0)
# Expected precision is 0 under the chosen undefined-denominator policy.
CLASS_SUPPORT = {0: 2, 1: 2}
