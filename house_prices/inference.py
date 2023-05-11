import numpy as np
from sklearn.impute import SimpleImputer
import joblib


def make_predictions(input_data):
    # Load the trained model and preprocessing objects
    model = joblib.load('../models/model.joblib')
    scaler = joblib.load('../models/scaler.joblib')
    encoder = joblib.load('../models/encoder.joblib')

    categorical_features = ['Foundation', 'KitchenQual']
    continuous_features = ['TotRmsAbvGrd', 'WoodDeckSF', 'YrSold', '1stFlrSF']

    imputer = SimpleImputer(strategy='most_frequent')
    input_data.loc[:, categorical_features] = \
        imputer.fit_transform(input_data.loc[:, categorical_features])

    input_data.loc[:, continuous_features] = \
        scaler.transform(input_data.loc[:, continuous_features])
    input_data_processed = encoder.transform(
        input_data.loc[:, categorical_features]
    )

    predictions = model.predict(
        np.hstack(
            (
                input_data.loc[:, continuous_features],
                input_data_processed.toarray()
            )
        )
    )

    return predictions
