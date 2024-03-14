This repository contains a Jupyter Notebook (Task_2.ipynb) that explores and classifies the Breast Cancer Wisconsin (Diagnostic) dataset. The task is to predict whether a breast mass is malignant or benign based on features extracted from mammogram images.

Goals:

    Explore and understand the features of the Breast Cancer Wisconsin (Diagnostic) dataset.
    Build and evaluate machine learning models for classifying breast cancer diagnoses (malignant vs. benign).

Data:

The Breast Cancer Wisconsin (Diagnostic) dataset
Notebook Contents:

    Data Loading and Cleaning
        Loads the dataset from a CSV file.
        Performs basic data cleaning (handling missing values, removing unnecessary columns).
        Standardizes features using StandardScaler.

    Data Exploration
        Analyzes data distribution (descriptive statistics, boxplots).
        Visualizes correlations between features using heatmaps.
        Explores the distribution of diagnoses ("Benign" vs. "Malignant").

    Model Building and Evaluation
        Trains and evaluates three classification models:
            Logistic Regression
            Support Vector Machine (SVM)
            Random Forest
        Uses GridSearchCV for basic hyperparameter tuning with Logistic Regression (can be extended to other models).
        Evaluates model performance using accuracy, precision, recall, and F1-score.
        Plots confusion matrices to visualize model predictions.

    Future Considerations
        Feature importance analysis to identify the most relevant features.
        Exploring class imbalance (if present) and handling techniques.
        Trying alternative classification algorithms (XGBoost, neural networks).

Running the Notebook on Google Colab

    Go to Colab: https://colab.research.google.com/.
    Upload the Task_2.ipynb notebook to your Colab workspace.
    Change the file path in the df = pd.read_csv(...) line to point to the location of your data file.
    Run the notebook cells (you can use "Run All" to execute all cells).

Dependencies:

This notebook requires the following Python libraries:

    pandas
    numpy
    scipy
    matplotlib
    seaborn
    sklearn (including sub-libraries like train_test_split, StandardScaler, LogisticRegression, SVC, RandomForestClassifier, GridSearchCV)

License:

This project is distributed under the MIT license. See the LICENSE file for details.
