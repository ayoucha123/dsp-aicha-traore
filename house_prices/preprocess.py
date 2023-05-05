import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_data(filepath):
    df = pd.read_csv('C:\Users\Matec Pro\dsp-aicha-traore\data\train.csv')
    return df

def preprocess_data(df):
    df = df.drop(columns=['Id'])
    df['LotFrontage'] = df['LotFrontage'].fillna(df['LotFrontage'].median())
    df = pd.get_dummies(df)
    return df

def scale_data(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled
