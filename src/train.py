def train_model(model, X_train, y_train):
    model = model.fit(X_train,y_train)

    return model