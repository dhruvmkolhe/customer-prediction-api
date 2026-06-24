import os
import sys
import json

# Add backend to path
sys.path.append(os.getcwd())

from predictor import Predictor

def test_importance():
    MODELS_DIR = os.path.dirname(__file__)
    print(f"Testing Predictor in {os.getcwd()}...")
    
    predictor = Predictor(MODELS_DIR)
    if not predictor.models_loaded:
        print("Models not loaded. Exiting.")
        return

    print(f"Loaded models: {list(predictor.classifiers.keys())}")
    
    for model_name in predictor.classifiers:
        print(f"\nTesting model: {model_name}")
        try:
            importance = predictor.get_feature_importance(model_name)
            print("Successfully retrieved importance:")
            print(json.dumps(importance, indent=2))
        except Exception as e:
            import traceback
            print(f"Error testing {model_name}:")
            traceback.print_exc()

if __name__ == "__main__":
    test_importance()
