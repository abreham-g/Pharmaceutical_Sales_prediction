from pathlib import Path


class Config:
  RANDOM_SEED = 50
  ASSETS_PATH = Path("../")
  REPO = "~/Desktop/10acad/abtest-mlops"
  DATASET_FILE_PATH = "Data/AdSmartdata.csv"
  DATASET_PATH = ASSETS_PATH / "Data"
  FEATURES_PATH = ASSETS_PATH / "features"
  MODELS_PATH = ASSETS_PATH / "models"
  METRICS_FILE_PATH = ASSETS_PATH / "metrics"
