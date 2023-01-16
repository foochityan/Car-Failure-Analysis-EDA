from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

class Config:
    # Data Processing parameters
    scale = True
    transform = True
    balance = True

    # Model training parameters
    cv = 5
    perform_cv = True
    return_top_models = 3

    # Model tuning parameter
    random_tune = True

    # Recommended models
    models = {
        "Random Forest": RandomForestClassifier(),
        "XGBoost": GradientBoostingClassifier(),
        "Logistic Regression": LogisticRegression()
    }

    # Base models
    base_models = {
        "Logistic Regression": LogisticRegression(),
        "SVM": SVC(),
        "Gaussian Naive Bayes": GaussianNB(),
        "K Nearest Neighbours": KNeighborsClassifier(),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": GradientBoostingClassifier()
    }

    # Model parameters
    model_params = {
        "Logistic Regression": {
            "penalty": ["none", "l2"]
        },
        "SVM": {
            "degree": list(range(1, 6))
        },
        "K Nearest Neighbours": {
            "n_neighbours": list(range(3, 11)),
            "weights": ["uniform", "distance"],
            "p": [1, 2, 3, 4]
        },
        "Decision Tree": {
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10]
        },
        "Random Forest": {
            "n_estimators": [50, 100, 150],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10]
        },
        "XGBoost": {
            "n_estimators": [50, 100, 150],
            "max_depth": [5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 5, 10]
        }
    }
