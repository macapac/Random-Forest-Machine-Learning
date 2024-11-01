import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score
import geopandas as gpd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# Load the CSV dataset
data = pd.read_csv(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\Project_datasets\data_index_2.csv')

# Load the world map
world = gpd.read_file(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\110m_cultural')

# Remove duplicate rows (if any)
data_cleaned = data.drop_duplicates()

def import_focus_region(data_cleaned, region_name):
     return data_cleaned[data_cleaned['Pan_2007']==region_name]

# Select only South America and Africa
data_americas = import_focus_region(data_cleaned, "'Americas'")
data_africa = import_focus_region(data_cleaned, "'Africa'")

# optional, plot the focus regions on the world map
def plot_focus_regions():
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot country boundaries
    world.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)

    # Plot misclassified points, color-coded by their actual class
    americas_gdf = gpd.GeoDataFrame(
        data_americas,
        geometry=gpd.points_from_xy(data_americas['Lon'], data_americas['Lat']),
        crs='EPSG:4326'
    )

    africa_gdf = gpd.GeoDataFrame(
        data_africa,
        geometry=gpd.points_from_xy(data_africa['Lon'], data_africa['Lat']),
        crs='EPSG:4326'
    )

    # Plot misclassified points on the same axis, using the 'Actual' column to color-code
    americas_gdf.plot(ax=ax, column='Biome_obs', cmap='viridis', markersize=10)
    africa_gdf.plot(ax=ax, column='Biome_obs', cmap='viridis', markersize=10)

    # Set plot title and labels
    ax.set_title('Focus regions')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Set tight layout and show the plot
    plt.tight_layout()
    plt.show()

# plot_focus_regions()

def split(data, test_size):
    X = data.drop(['CN','pH','cellfraction','NPP' ,'SoilR' ,'MaxBiomeCmax',
    'MaxBiomeLAI', 'VegC', 'LitterC','SoilC' ,'Biome_Cmax', 'Biome_LAI',
    'Biome_obs','GFED-region', 'Pan_2007','ISO3', 'UN'], axis=1)

    y_obs = data['Biome_obs']
    y_cmax = data['Biome_Cmax']

    # Perform the train-test split for X, y_obs, and y_cmax simultaneously
    X_train, X_test, y_obs_train, y_obs_test, y_cmax_train, y_cmax_test = train_test_split(
        X, y_obs, y_cmax, test_size=test_size, random_state=42
    )

    return X_train, X_test, y_obs_train, y_obs_test, y_cmax_train, y_cmax_test

X_americas_train, X_americas_test,  obs_americas_train, obs_americas_test, cmax_americas_train, cmax_americas_test = split(data_americas, test_size=0.2)
X_africa_train, X_africa_test,  obs_africa_train, obs_africa_test, cmax_africa_train, cmax_africa_test = split(data_africa, test_size=0.2)

def train_random_forest(X_train, y_obs_train, y_cmax_train):
    model_obs = RandomForestClassifier(n_estimators=100, random_state=42)
    model_obs.fit(X_train, y_obs_train)

    model_cmax = RandomForestClassifier(n_estimators=100, random_state=42)
    model_cmax.fit(X_train, y_cmax_train)

    return model_obs, model_cmax

obs_americas_model, cmax_americas_model = train_random_forest(X_americas_train, obs_americas_train, cmax_americas_train)
obs_africa_model, cmax_africa_model = train_random_forest(X_africa_train, obs_africa_train, cmax_africa_train)

def evaluate_random_forest(model_obs, model_cmax, X_test, obs_test, cmax_test, should_plot_confusion_matrix=False, region_name = ''):
    obs_pred = model_obs.predict(X_test)
    cmax_pred = model_cmax.predict(X_test)

    def print_metrics(y_true, y_pred, label):
        print(f"\nClassification Report {label}:\n", classification_report(y_true, y_pred, zero_division=1))

        # Additional Metrics for Random Forest (Multiclass Classification)
        f1_rf = f1_score(y_true, y_pred, average='weighted')  # Use 'weighted' for multiclass
        mcc_rf = matthews_corrcoef(y_true, y_pred)  # MCC works for multiclass
        kappa_rf = cohen_kappa_score(y_true, y_pred)  # Cohen's Kappa works for multiclass

        print(f"F1 Score ({label}):", f1_rf)
        print(f"Matthews Correlation Coefficient ({label}):", mcc_rf)
        print(f"Cohen's Kappa ({label}):", kappa_rf, "\n")

    # Use the function for both models' predictions
    print_metrics(obs_test, obs_pred, "Biome_obs")
    print_metrics(cmax_test, cmax_pred, "Biome_Cmax")

    if should_plot_confusion_matrix == True:
        def plot_confusion_matrix(y_true, y_pred, label):
            cm = confusion_matrix(y_true, y_pred)
            plt.figure(figsize=(10, 7))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {label} - {region_name}')
            plt.ylabel('Actual Class')
            plt.xlabel('Predicted Class')
            plt.show()

        plot_confusion_matrix(obs_test, obs_pred, "Biome_obs")
        plot_confusion_matrix(cmax_test, cmax_pred, "Biome_Cmax")

evaluate_random_forest(obs_americas_model, cmax_americas_model, X_americas_test, obs_americas_test, cmax_americas_test, True, 'Americas')
evaluate_random_forest(obs_africa_model, cmax_africa_model, X_africa_test, obs_africa_test, cmax_africa_test, True, 'Africa')

def tune(X_train, y_obs_train, y_cmax_train):
    print('\nTuning Hyperparameters\n')
    # Define the hyperparameter grid
    param_dist = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'bootstrap': [True, False]
    }

    # Implement RandomizedSearchCV
    random_search_obs = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=40,  # Number of parameter settings sampled
        cv=5,  # 5-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    random_search_obs.fit(X_train, y_obs_train)
    best_obs_model = random_search_obs.best_estimator_
    print('Obs done')

    # Implement RandomizedSearchCV
    random_search_cmax = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_distributions=param_dist,
        n_iter=40,  # Number of parameter settings sampled
        cv=5,  # 5-fold cross-validation
        verbose=1,
        random_state=42,
        n_jobs=-1  # Use all available cores
    )

    random_search_cmax.fit(X_train, y_cmax_train)
    best_cmax_model = random_search_cmax.best_estimator_
    print('Cmax done')

    return best_obs_model, best_cmax_model

best_obs_americas_model, best_cmax_americas_model = tune(X_americas_train, obs_americas_train, cmax_americas_train)
best_obs_africa_model, best_cmax_africa_model = tune(X_africa_train, obs_africa_train, cmax_africa_train)

evaluate_random_forest(best_obs_americas_model, best_cmax_americas_model, X_americas_test, obs_americas_test, cmax_americas_test, True, 'Americas tuned')
evaluate_random_forest(best_obs_africa_model, best_cmax_africa_model, X_africa_test, obs_africa_test, cmax_africa_test, True, 'Africa tuned')

# Evaluate tuned Americas model on Africa data
print("\nEvaluating Tuned Americas-trained model on Africa test data:")
evaluate_random_forest(best_obs_americas_model, best_cmax_americas_model, X_africa_test, obs_africa_test, cmax_africa_test, True, 'Tuned Americas Model on Africa')

# Evaluate tuned Africa model on Americas data
print("\nEvaluating Tuned Africa-trained model on Americas test data:")
evaluate_random_forest(best_obs_africa_model, best_cmax_africa_model, X_americas_test, obs_americas_test, cmax_americas_test, True, 'Tuned Africa Model on Americas')

def plot_feature_importance(model, X_train):
    # Compute feature importances from the best random search model
    importances = model.feature_importances_

    # Get the feature names
    feature_names = X_train.columns

    # Create a DataFrame for the feature importances
    feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

    # Sort the DataFrame by importance
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

    # Plot the top 10 most important features
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
    plt.title('Top 10 Feature Importances')
    plt.show()

plot_feature_importance(best_obs_americas_model, X_americas_train)
plot_feature_importance(best_cmax_americas_model, X_americas_train)
plot_feature_importance(best_obs_africa_model, X_africa_train)
plot_feature_importance(best_cmax_africa_model, X_africa_train)
