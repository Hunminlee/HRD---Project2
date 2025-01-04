from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier, plot_importance

def train_test_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train, verbose=False)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy (Same year) ========> ", accuracy)

    return model


def Grid_search(model, X_train, y_train):

    # Best Parameters: {'colsample_bytree': 0.6, 'gamma': 0.1, 'learning_rate': 0.01, 'max_depth': 7, 'n_estimators': 300, 'subsample': 0.8}

    model = XGBClassifier()
    # subsample=0.8, reg_lambda=0, reg_alpha=0.5, n_estimators=200,
    # min_child_weight=3, max_depth=7, learning_rate=0.1, gamma=0.1,
    # colsample_bytree=0.6
    # )

    param_grid = {
        'learning_rate': [0.01, 0.05, 0.1],  # Step size shrinking
        'n_estimators': [100, 200, 300],  # Number of boosting rounds
        'max_depth': [3, 5, 7, 10],  # Depth of each tree
        'subsample': [0.7, 0.8, 1.0],  # Fraction of samples for training each tree
        'colsample_bytree': [0.6, 0.7, 0.8],  # Fraction of features for each tree
        'gamma': [0.0, 0.1, 0.2]  # Regularization term for pruning
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='accuracy',
        cv=3,
        n_jobs=-1,
        verbose=1
    )

    # 5분 정도 걸림
    grid_search.fit(X_train, y_train)

    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)