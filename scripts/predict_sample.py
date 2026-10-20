"""Call the running local API with a sample payload; does not make medical claims."""

import argparse
import json

import requests

SAMPLE = {
    "age": 57, "sex": 1, "cp": 2, "trestbps": 130, "chol": 236,
    "fbs": 0, "restecg": 0, "thalach": 174, "exang": 0,
    "oldpeak": 0.0, "slope": 2, "ca": 0, "thal": 3,
}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    args = parser.parse_args()
    response = requests.post(args.url, json=SAMPLE, timeout=20)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))

if __name__ == "__main__":
    main()
