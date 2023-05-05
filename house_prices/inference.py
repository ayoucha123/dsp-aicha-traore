import pandas as pd

def make_predictions(model, X):
    y_pred = model.predict(X)
    return y_pred

def save_predictions(y_pred, filepath):
    df = pd.DataFrame({'SalePrice': y_pred})
    df.to_csv(filepath, index=False)
