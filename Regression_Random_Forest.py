import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
import geopandas as gpd
from sklearn.model_selection import train_test_split

# Load the CSV dataset
data = pd.read_csv(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\Project_datasets\data_index_2.csv')

# Load the LPJ guess output
LPJ_guess_data = pd.read_csv(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\Project_datasets\LPJ-GUESS_output_BERN1.csv')

# Remove duplicate rows (if any)
data_cleaned = data.drop_duplicates()

def split(data, test_size):
    X = data.drop(['CN','pH','cellfraction','NPP' ,'SoilR' ,'MaxBiomeCmax',
    'MaxBiomeLAI', 'VegC', 'LitterC','SoilC' ,'Biome_Cmax', 'Biome_LAI',
    'Biome_obs','GFED-region', 'Pan_2007','ISO3', 'UN'], axis=1)

    y_npp = data['NPP']
    y_vegc = data['VegC']

    # Perform the train-test split for X, y_npp, and y_vegc simultaneously
    X_train, X_test, y_npp_train, y_npp_test, y_vegc_train, y_vegc_test = train_test_split(
        X, y_npp, y_vegc, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_npp_train, y_npp_test, y_vegc_train, y_vegc_test

X_train, X_test, y_npp_train, y_npp_test, y_vegc_train, y_vegc_test = split(data_cleaned, test_size=0.2)
print("Data Splitting done")

def train_random_forest(X_train, y_npp_train, y_vegc_train):
    model_npp = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1, verbose=1)
    model_npp.fit(X_train, y_npp_train)

    model_vegc = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1, verbose=1)
    model_vegc.fit(X_train, y_vegc_train)

    return model_npp, model_vegc

npp_model, vegc_model = train_random_forest(X_train, y_npp_train, y_vegc_train)
print("Training Done")

def evaluate_random_forest(model_npp, model_vegc, X_test, npp_test, vegc_test, should_plot_scatter=False):
    npp_pred = model_npp.predict(X_test)
    vegc_pred = model_vegc.predict(X_test)

    def print_metrics(y_true, y_pred, label):
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)


        # Print the regression metrics
        print(f"\nEvaluation for {label} Regression Model:")
        print("Mean Absolute Error (MAE):", mae)
        print("Mean Squared Error (MSE):", mse)
        print("Root Mean Squared Error (RMSE):", rmse)
        print("R^2 Score:", r2)

    # Use the function for both models' predictions
    print_metrics(npp_test, npp_pred, "NPP")
    print_metrics(vegc_test, vegc_pred, "VegC")

    if should_plot_scatter:
        def plot_true_vs_predicted(y_true, y_pred, label):
            plt.figure(figsize=(8, 8))
            plt.scatter(y_true, y_pred, alpha=0.6, edgecolors='k', s=50)

            # Plot the y=x line for reference
            min_val = min(y_true.min(), y_pred.min())
            max_val = max(y_true.max(), y_pred.max())
            plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal: y=x")  # Reference line y=x

            # Set plot details
            plt.title(f"True vs. Predicted Values - {label}")
            plt.xlabel('True Values')
            plt.ylabel('Predicted Values')
            plt.legend()
            plt.grid(True)
            plt.show()

        plot_true_vs_predicted(npp_test, npp_pred, "NPP")
        plot_true_vs_predicted(vegc_test, vegc_pred, "VegC")

evaluate_random_forest(npp_model, vegc_model, X_test, y_npp_test, y_vegc_test, True)

def plot_feature_importance(model, X_train):

    importances = model.feature_importances_

    # Get the feature names
    feature_names = X_train.columns

    # Create a DataFrame for the feature importances
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the top 10 most important features
    plt.figure(figsize=(12, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.show()

plot_feature_importance(npp_model, X_train)
plot_feature_importance(vegc_model, X_train)

def compare_to_lpj(model_npp, model_vegc, X_test, lpj_data):
    lpj_matched = lpj_data.merge(X_test[['Lon', 'Lat']], on=['Lon', 'Lat'], how='inner')
    lpj_matched = lpj_matched.set_index(['Lon', 'Lat']).reindex(X_test.set_index(['Lon', 'Lat']).index).reset_index()  # Making sure the lpj output is in the same order

    npp_pred = model_npp.predict(X_test)
    vegc_pred = model_vegc.predict(X_test)

    def plot_scatter_with_xy_line(y_1, y_2, title, xlabel, ylabel):
        plt.figure(figsize=(8,8))
        plt.scatter(y_1, y_2, alpha=0.6, edgecolors='k', s=50, label="Predictions")

        # Plot the y=x line for reference
        min_val = min(y_1.min(), y_2.min())
        max_val = max(y_1.max(), y_2.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal: y=x")

        # Set plot details
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.legend()
        plt.grid(True)
        plt.show()

    plot_scatter_with_xy_line(lpj_matched['NPP'], npp_pred, "NPP from LPJ guess against machine learning", "LPJ guess", "Random forest")
    plot_scatter_with_xy_line(lpj_matched['VegC'], vegc_pred, "VegC from LPJ guess against machine learning", "LPJ guess","Random forest")

compare_to_lpj(npp_model, vegc_model, X_test, LPJ_guess_data)
