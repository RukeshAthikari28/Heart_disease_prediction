import os
import sys
from dataclasses import dataclass
import numpy as np

from catboost import CatBoostClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
      try:
        logging.info("Split training and test input data")
        X_train, y_train, X_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],   
            test_array[:, -1],
        )
        y_train = y_train.astype(int)
        y_test = y_test.astype(int)
        # Assuming y_train and y_test are your target labels
        y_train = y_train.replace(2, 1)
        y_test = y_test.replace(2, 1)

 

        logging.info(f"Training features shape: {X_train.shape}, Training target shape: {y_train.shape}")
        logging.info(f"Testing features shape: {X_test.shape}, Testing target shape: {y_test.shape}")

    

        models = {
            "Random Forest": RandomForestClassifier(),
            "Decision Tree": DecisionTreeClassifier(),
            "Gradient Boosting": GradientBoostingClassifier(),
            "Logistic Regression": LogisticRegression(),
            "XGBClassifier": XGBClassifier(),
            "CatBoost Classifier": CatBoostClassifier(verbose=False),
            "AdaBoost Classifier": AdaBoostClassifier(),
        }

        model_report: dict = evaluate_models(
            X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models,param=None
        )

        # Get best model score from the report
        best_model_score = max(sorted(model_report.values()))

        # Get best model name from the report
        best_model_name = list(model_report.keys())[
            list(model_report.values()).index(best_model_score)
        ]
        best_model = models[best_model_name]

        if best_model_score < 0.6:
            raise CustomException("No best model found")

        logging.info(f"Best found model on both training and testing datasets: {best_model_name}")

        save_object(
            file_path=self.model_trainer_config.trained_model_file_path,
            obj=best_model,
        )

        predicted = best_model.predict(X_test)

        accuracy = accuracy_score(y_test, predicted)
        
        return accuracy

      except Exception as e:
          raise CustomException(e, sys)
