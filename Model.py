from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, plot_importance
#from lightgbm import LGBMClassifier, plot_importance
#from catboost import CatBoostClassifier
#from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import pandas as pd


def build_model():
    model = XGBClassifier(
        subsample=0.8, reg_lambda=0, reg_alpha=0.5, n_estimators=300,
        min_child_weight=3, max_depth=7, learning_rate=0.1, gamma=0.1,
        colsample_bytree=0.6
    )

    return model




def feature_importance_save(model, idx1):
    importance = model.get_booster().get_score(importance_type='weight')

    sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    importance_df = pd.DataFrame(sorted_importance, columns=['Feature', 'Importance'])
    importance_df.to_csv(f'./Feature_importance_202{idx1}.csv', index=False)


def permutation_importance_save(model, X_test, y_test, idx1, idx2):
    result = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)

    importance_df = pd.DataFrame({
        'Feature': X_test.columns,
        'Importance': result.importances_mean
    })

    importance_df_sorted = importance_df.sort_values(by='Importance', ascending=False)

    # top_10_features = importance_df_sorted.head(10)
    # print("Top 10 Important Features:")
    # print(top_10_features)

    # plt.barh(top_10_features['Feature'], top_10_features['Importance'])
    # plt.title("Top 10 Permutation Feature Importance")
    # plt.xlabel("Importance")
    # plt.show()
    importance_df_sorted.to_csv(f'./Permutation_importance_Train_202{idx1}_Test_202{idx2}.csv', index=False)