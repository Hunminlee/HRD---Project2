import Model
import data_preprocessing as DP
import utils
import evaluation
import numpy as np
#import config
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")



def main():
    if __name__ == "__main__":

        for i in range(2020, 2024):
            globals()[f'Work_data_{i}'] = DP.call_X_data(i, 'Work')
            globals()[f'Head_data_{i}'] = DP.call_X_data(i, 'Head')
            print(globals()[f'Work_data_{i}'].shape, globals()[f'Head_data_{i}'].shape)

        path = './SPSS/'
        file_path = path + 'Y_label.xlsx'  # Original dataset model with different name for flexible data input

        var = pd.read_excel(file_path, sheet_name='Items', engine='openpyxl')

        target_full_name = list(var.iloc[1, :])[2]
        target_2020 = list(var.iloc[1, :])[3]
        target_2021 = list(var.iloc[1, :])[4]
        target_2022 = list(var.iloc[1, :])[5]
        target_2023 = list(var.iloc[1, :])[6]

        # X_df = var.iloc[2:, :] #All features
        X_df = var[var['분류'] == 'Feature_2']  # Selected features Only
        # X_df = var[var['분류'] == 'Feature_3' or var['분류'] == 'Feature_3']   # Selected features Only
        '''
        ['Q10A', 'Q10B', 'Q10C', 'Q10D', 'Q11A1', 'Q11B1', 'Q11A2', 'Q11B2', 'Q12A', 'Q12B', 'Q12C', 'Q13', 'Q14A', 'Q14B', 'Q14C', 'Q14D',
         'Q14E', 'Q14F', 'Q15A1', 'Q15B1', 'Q15C1', 'Q15D1', 'Q15E1', 'Q15A2', 'Q15B2', 'Q15C2', 'Q15D2', 'Q15E2', 'Q16A1', 'Q16B1', 'Q16C1',
         'Q16D1', 'Q16A2', 'Q16B2', 'Q16C2', 'Q16D2', 'Q17']
       '''


        lst_data, cnt_data = [], []

        for i in range(2020, 2024):

            if i==2020: var_names = X_df['변수명'].tolist()
            elif i == 2021: var_names = X_df['Unnamed: 4'].tolist()
            elif i == 2022: var_names = X_df['Unnamed: 5'].tolist()
            elif i == 2023: var_names = X_df['Unnamed: 6'].tolist()

            globals()[f'X_{i}'] = globals()[f'Work_data_{i}'][var_names]

            print(f"{i} Dataset shape : ", globals()[f'X_{i}'].shape)
            globals()[f'Nan_cols_{i}'], globals()[f'cnt_{i}'] = DP.count_nan_and_extract_cols(
                globals()[f'X_{i}'].isna().sum())

            lst_data.append(globals()[f'Nan_cols_{i}'])
            cnt_data.append(globals()[f'cnt_{i}'])

        utils.draw_fig_check_nan(lst_data, cnt_data)

        Nan_col_check_2020 = [item.replace('W20', '') for item in Nan_cols_2020]
        Nan_col_check_2021 = [item.replace('W21', '') for item in Nan_cols_2021]
        Nan_col_for_2022 = [item.replace('W20', 'W22') for item in Nan_cols_2020]  # to maintain coherence
        Nan_col_check_2023 = [item.replace('W23', '') for item in Nan_cols_2023]

        tmp = Nan_col_check_2020 == Nan_col_check_2021 == Nan_col_check_2023
        if tmp:
            print("Check values - should be True :", tmp)  # Identical columns that involves Nan
        else:
            raise ValueError("Columns are not identical or contain different NaN values.")

        for i in range(2020, 2024):
            globals()[f'X_{i}_cleaned'] = globals()[f'X_{i}'].drop(columns=globals()[f'Nan_cols_{i}'])
            if i == 2022:
                globals()[f'X_{i}_cleaned'] = globals()[f'X_{i}'].drop(columns=Nan_col_for_2022)

            print(f"Dataset {i}'s Nan value: ", globals()[f'X_{i}_cleaned'].isna().sum().sum(),
                  f"and shape : {globals()[f'X_{i}_cleaned'].shape}")
        # X_2020_cleaned

        utils.draw_fig_class_check([Work_data_2020, Work_data_2021, Work_data_2022, Work_data_2023],
                             [target_2020, target_2021, target_2022, target_2023])

        for i in range(4):
            if i==0: tmp_idx = target_2020
            elif i == 1: tmp_idx = target_2021
            elif i == 2: tmp_idx = target_2022
            elif i == 3: tmp_idx = target_2023

            tmp = globals()[f'Work_data_202{i}'][tmp_idx]
            tmp_lst = []

            for v in tmp:
                if v == 1 or v == 2:
                    tmp_lst.append(0)
                elif v == 4 or v == 5:
                    tmp_lst.append(1)
                else:
                    tmp_lst.append(2)

            globals()[f'Y_202{i}'] = pd.Series(tmp_lst)

        utils.draw_fig_class([Y_2020, Y_2021, Y_2022, Y_2023])

        for i in range(2020, 2024):
            y_df = pd.DataFrame(globals()[f'Y_{i}'], columns=['Target'])

            globals()[f'final_data_{i}'] = pd.concat([globals()[f'X_{i}_cleaned'], y_df], axis=1)
            num_val = globals()[f'final_data_{i}']['Target'].value_counts().values[-1]

            class_0_samples = globals()[f'final_data_{i}'][globals()[f'final_data_{i}']['Target'] == 0].sample(
                n=num_val, random_state=42)
            class_1_samples = globals()[f'final_data_{i}'][globals()[f'final_data_{i}']['Target'] == 1].sample(
                n=num_val, random_state=42)

            selected_data = pd.concat([class_0_samples, class_1_samples], axis=0).reset_index(drop=True)

            globals()[f'selected_data_{i}'] = selected_data

        utils.draw_fig_class([selected_data_2020['Target'], selected_data_2021['Target'], selected_data_2022['Target'],
                        selected_data_2023['Target']])

        scaler = StandardScaler()

        for i in range(4):
            X = globals()[f'selected_data_202{i}'].drop(columns=['Target'])
            # X = pd.DataFrame(scaler.fit_transform(X_unnorm), columns=X_unnorm.columns)
            X.columns = X.columns.str.replace(f'W2{i}', '', regex=False)  ##Important for generalization
            Y = globals()[f'selected_data_202{i}']['Target']

            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

            DP.Check_String(X_train)
            DP.Check_String(X_test)

            model = Model.build_model()
            model = evaluation.train_test_model(model, X_train, y_train, X_test, y_test)
            Model.feature_importance_save(model, i)

            for j in range(i, 4):
                X = globals()[f'selected_data_202{j}'].drop(columns=['Target'])
                # X = pd.DataFrame(scaler.fit_transform(X_unnorm), columns=X_unnorm.columns)
                X.columns = X.columns.str.replace(f'W2{j}', '', regex=False)  ##Important for generalization
                Y = globals()[f'selected_data_202{j}']['Target']

                X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

                DP.Check_String(X_train)
                DP.Check_String(X_test)

                y_pred = model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                print(f"Model trained with Year 202{i}, testing with data from year 202{j} => Accuracy : {accuracy}")

                Model.permutation_importance_save(model, X_test, y_test, i, j)


if __name__ == "__main__":
    main()