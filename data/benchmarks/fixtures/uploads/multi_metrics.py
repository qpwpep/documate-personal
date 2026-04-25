from sklearn.metrics import accuracy_score, f1_score

def evaluate_predictions(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    return {'accuracy': accuracy, 'macro_f1': macro_f1}
