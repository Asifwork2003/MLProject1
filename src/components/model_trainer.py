import os
import sys
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score
from xgboost import XGBRegressor
from src.utils import save_object, evaluate_model
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params = {
                "Decision Tree": {'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']},
                "Random Forest": {'n_estimators': [8, 16, 32, 64, 128, 256]},
                "Gradient Boosting": {'learning_rate': [0.1, 0.01, 0.05, 0.001], 'subsample': [0.6, 0.7, 0.75, 0.8, 0.85, 0.9], 'n_estimators': [8, 16, 32, 64, 128, 256]},
                "XGBRegressor": {'learning_rate': [0.1, 0.01, 0.05, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]},
                "CatBoosting Regressor": {'depth': [6, 8, 10], 'learning_rate': [0.01, 0.05, 0.1], 'iterations': [30, 50, 100]},
                "AdaBoost Regressor": {'learning_rate': [0.1, 0.01, 0.5, 0.001], 'n_estimators': [8, 16, 32, 64, 128, 256]}
            }
            
            model_report = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, param=params)
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]
            best_score = model_report[best_model_name]

            if best_score < 0.7:
                raise CustomException("No suitable model found")
                
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            return best_score

        except Exception as e:
            raise CustomException(e, sys)
