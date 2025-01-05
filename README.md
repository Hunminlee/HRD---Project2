# HRD Environment Estimation Using HCCP Dataset 

## Project Overview 

This project aims to estimate various aspects of the **HRD (Human Resource Development) environment** using the **HCCP dataset**. The dataset is publicly available from the **KRIVET** (Korea Research Institute for Vocational Education and Training) website [here](https://www.krivet.re.kr/kor/sub.do?menuSn=21). 
The goal of the project is to develop a model that can help predict important aspects of HRD environments, leveraging the data provided in the HCCP dataset. 

This involves data preprocessing, model training, evaluation, and feature importance analysis, with the final objective of delivering a predictive model suitable for HRD-related applications. 

## Project Structure 
The project is organized into the following structure: 
```
my_project/
├── main.py # Main script that runs the overall pipeline
├── data_preprocessing.py # Functions related to data loading, cleaning, and preprocessing
├── model.py # Model building and training functions
├── evaluation.py # Evaluation and performance functions (accuracy, feature importance, etc.)
├── utils.py # Utility functions like checks and plotting
└── requirements.txt # Dependencies file
``` 

### File Descriptions 
1. **`main.py`** This is the main entry point for the project. It runs the entire pipeline, including data loading, preprocessing, model training, and evaluation. It also integrates various functions from other modules to produce the final results. 
2. **`data_preprocessing.py`** This file contains all functions related to data handling, including loading datasets, cleaning missing values, performing feature selection, and transforming the data into a format suitable for modeling. 
3. **`model.py`** The model building and training functions reside here. It includes code to define, train, and evaluate machine learning models (such as classification or regression models). Hyperparameter tuning and model optimization may also be included in this file. 
4. **`evaluation.py`** This file contains functions related to model evaluation. It includes metrics such as accuracy, precision, recall, and others. It also includes code for extracting feature importance and visualizing the performance of the model.
5. **`utils.py`** Utility functions such as plotting, data checks, and helper functions are located here. These functions help in performing common tasks such as visualizing data, checking for missing values, or any other repetitive operations needed throughout the project.
6. **`requirements.txt`** This file lists all the dependencies required for the project. To install the required libraries, run: ``` pip install -r requirements.txt ```

## Requirements To set up the project, you'll need to install the following dependencies (listed in `requirements.txt`)
You can install them all by running the following command: ``` pip install -r requirements.txt ``` 

## How to Use 
1. Clone the repository: ``` git clone https://github.com/Hunminlee/HRD---Project2.git cd my_project ```
2. Install the required dependencies: ``` pip install -r requirements.txt ```
3. Run the project: To execute the full pipeline, run the `main.py` script: ``` python main.py ``` This will automatically load the data, preprocess it, train the model, and display evaluation results.

## License This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details. 

## Acknowledgments - The HCCP dataset is provided by the **Korea Research Institute for Vocational Education and Training (KRIVET)**. 
