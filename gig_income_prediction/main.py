import os
from utils.io_utils import load_data, save_model
from features.feature_engineering import create_features
from models.pipeline import create_pipeline
from models.train import train_models
from models.evaluate import evaluate_models
from utils.config import DATA_PATH, MODEL_OUTPUT_PATH

def main():
    # Load the data
    raw_data = load_data(os.path.join(DATA_PATH, 'raw_data.csv'))
    
    # Create features
    features = create_features(raw_data)
    
    # Create the model pipeline
    pipeline = create_pipeline()
    
    # Train models
    trained_models = train_models(pipeline, features)
    
    # Evaluate models
    evaluation_results = evaluate_models(trained_models, features)
    
    # Save the trained models
    for model_name, model in trained_models.items():
        save_model(model, os.path.join(MODEL_OUTPUT_PATH, f"{model_name}.pkl"))
    
    print("Training and evaluation completed successfully.")

if __name__ == "__main__":
    main()