import yaml
from src.linear_regression import LinearRegression
from src.train import train_model
from src.test import test_model
from src.preprocessing import run_preprocessing


def main():
    with open("config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    
    print("Loading data...\n")
    X_train, y_train, X_test, y_test, scaler = run_preprocessing(config)
    
    print("Initializing model...\n")
    model = LinearRegression()
    
    print("Training has started")
    model = train_model(model, X_train, y_train)

    print("Training has been completed successfully!\n")

    print("Evaluating model...\n")
    test_model(model, X_test, y_test, scaler)
    print("Evaluation completed!")
    
if __name__ == "__main__":
    main()