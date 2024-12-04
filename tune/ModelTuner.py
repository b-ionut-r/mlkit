from mlkit.linear_model import (LinearRegression, LogisticRegression, 
                                MultiLogisticRegression)
from mlkit.neighbors import KNeighborsClassifier
from mlkit.preprocessing import MinMaxScaler, StandardScaler
from mlkit.model_selection import KFold
import numpy as np
import optuna
from typing import Union
import contextlib
import os


import warnings
warnings.simplefilter("ignore", category=FutureWarning)
warnings.simplefilter("ignore", category=UserWarning)

# Configure a null handler for logging
import logging
logging.basicConfig(level=logging.INFO)
null_handler = logging.NullHandler()
logger = logging.getLogger("model_training")
logger.addHandler(null_handler)




class ModelTuner:

    def __init__(self, 
                 model: Union[LinearRegression, LogisticRegression, MultiLogisticRegression, 
                              KNeighborsClassifier], 
                 val_splitter = KFold(n_splits=5),
                 hyperparams_space = {
                    "n_iters": (1000, 15000, 2000, "int"), 
                    "lr": (1e-2, 0.1, "float_log"),       
                    "reg_mode": (["l1", "l2", "elasticnet"], "cat"),      
                    "reg_strength": (1e-3, 1e-2, "float_log"),
                    "reg_ratio": (0.2, 0.8, "float"),   
                    "method": (["sgd", "gd"], "cat"),          
                    "batch_size": ([128, 256], "cat"),  
                    "scaler": ([MinMaxScaler(), StandardScaler()], "cat"),    
                    "verbose": ([0], "cat"),
                 },
                 solver = "optuna",
                 n_trials = 10000,
                 timeout = 1800, # 30 min
                 n_jobs = 1,
                 show_progress_bar = True
                ):

        self.model = model
        self.val_splitter = val_splitter
        self.hyperparams_space = hyperparams_space
        self.solver = solver
        self.n_trials = n_trials 
        self.timeout = timeout
        self.n_jobs = n_jobs
        self.show_progress_bar = show_progress_bar
        self.study = None

    def fit(self, X, y):

        if self.solver == "optuna":
        
            def objective(trial):

                params = {}
                for key, value in self.hyperparams_space.items():
                    if value[-1] == "int":
                        params[key] = trial.suggest_int(key, value[0], value[1], step=value[2])
                    elif value[-1] == "float":
                        params[key] = trial.suggest_float(key, value[0], value[1])
                    elif value[-1] == "float_log":
                        params[key] = trial.suggest_float(key, value[0], value[1], log = True)
                    elif value[-1] == "cat":
                        params[key] = trial.suggest_categorical(key, value[0])

                if "reg_mode" in params:       
                    reg_mode, reg_strength, reg_ratio = params.pop("reg_mode"), params.pop("reg_strength"), params.pop("reg_ratio")
                    if reg_mode == "elasticnet":
                        params["reg"] = (reg_mode, reg_strength, reg_ratio)
                    else:
                        params["reg"] = (reg_mode, reg_strength)


                model = type(self.model)(**params)
                if isinstance(model, LinearRegression):
                    metric = "mae"
                elif isinstance(model, (LogisticRegression, KNeighborsClassifier)):
                    metric = "accuracy"
                elif isinstance(model, MultiLogisticRegression):
                    metric = "f1"


                self.val_splitter = type(self.val_splitter)(**vars(self.val_splitter))
                fold_metrics = []
                for train_idx, val_idx in self.val_splitter.split(X):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    model.fit(X_train, y_train)
                    fold_metric = model.score(X_val, y_val)[metric]
                    fold_metrics.append(fold_metric)

                if metric == "mae":
                    return -np.mean(fold_metrics)
                else:
                    return np.mean(fold_metrics)
            
            self.study = optuna.create_study(study_name = "tuner_study",
                                             direction = "maximize",
                                             sampler = optuna.samplers.TPESampler(),
                                             pruner = optuna.pruners.MedianPruner())
            

            print("\nStarting optimization process...\n")
            self.study.optimize(objective, n_trials = self.n_trials, timeout = self.timeout,
                                n_jobs = self.n_jobs, gc_after_trial = True, 
                                show_progress_bar = self.show_progress_bar)
            
            best_params = self.study.best_params
            if "reg_mode" in best_params:
                reg_mode, reg_strength, reg_ratio = (best_params.pop("reg_mode"), best_params.pop("reg_strength"), 
                                                    best_params.pop("reg_ratio"))
                if reg_mode == "elasticnet":
                    best_params["reg"] = (reg_mode, reg_strength, reg_ratio)
                else:
                    best_params["reg"] = (reg_mode, reg_strength)
            print(f"\nFinished tuning. Best params are:\n{best_params}")

            print("\nRetraining best model on entire dataset.")
            best_model = type(self.model)(**best_params)
            best_model.fit(X, y)
            print("Done. 😄")

            return best_model
        