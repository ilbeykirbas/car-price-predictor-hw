from src.metrics import summary
import numpy as np

def test_model(model, X_test, y_test, scaler):
    preds = model.predict(X_test)

    preds = scaler.inverse_transform(preds)
    y_test = scaler.inverse_transform(y_test)

    preds = np.expm1(preds)   # log'u geri al
    y_test = np.expm1(y_test) # log'u geri al

    errors = summary(y_test, preds)

    print(errors)