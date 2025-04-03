import os
from src.train import train
from src.web_service import launch


def main():
    if os.getenv("TRAINING_MODE", "OFF") == "ON":
        print("Training the model...")
        train()
    else:
        print("Starting the web service...")
        launch()


if __name__ == "__main__":
    main()
