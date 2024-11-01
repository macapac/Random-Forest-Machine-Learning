import pandas as pd
import numpy as np
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import fiona
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, matthews_corrcoef, cohen_kappa_score

# Load the CSV dataset
data = pd.read_csv(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\Project_datasets\data_index_2.csv')

# Display the first few rows of the dataset
print("Initial data sample:\n", data.head())

# Basic statistics for continuous variables
print("\nBasic statistics:\n", data.describe())

# Histogram for the biomes (Biome_obs)
plt.figure(figsize=(10, 6))
data['Biome_obs'].hist(bins=30, color='teal', edgecolor='black')
plt.title('Distribution of Biomes')
plt.xlabel('Biome Category')
plt.ylabel('Frequency')
plt.show()

# Boxplot for soil texture (clay, silt, sand)
data[['clay', 'silt', 'sand']].plot(kind='box', figsize=(10, 6), grid=True)
plt.title('Soil Texture Distribution')
plt.show()

# Check for missing values
missing_values = data.isnull().sum()
print("\nMissing values per column:\n", missing_values)

data_cleaned = data.drop_duplicates()

# Verify Biome categories
print("\nUnique biome categories:\n", data_cleaned['Biome_obs'].unique())

# For binary classification, selecting Tropical rain and seasonal forest in bolivia and brazil
data_bolivia = data_cleaned[(data_cleaned['Biome_obs'].isin([8, 9])) & (data_cleaned['UN']==68)]
data_brazil = data_cleaned[(data_cleaned['Biome_obs'].isin([8, 9])) & (data_cleaned['UN']==76)]

# Brazil is train and Bolivia is test
X_train = data_brazil.drop(['CN','pH','cellfraction','NPP' ,'SoilR' ,'MaxBiomeCmax',
'MaxBiomeLAI', 'VegC', 'LitterC','SoilC' ,'Biome_Cmax', 'Biome_LAI',
'Biome_obs','GFED-region', 'Pan_2007','ISO3', 'UN'], axis=1)
y_train = data_brazil['Biome_obs']

X_test = data_bolivia.drop(['CN','pH','cellfraction','NPP' ,'SoilR' ,'MaxBiomeCmax',
'MaxBiomeLAI', 'VegC', 'LitterC','SoilC' ,'Biome_Cmax', 'Biome_LAI',
'Biome_obs','GFED-region', 'Pan_2007','ISO3', 'UN'], axis=1)
y_test = data_bolivia['Biome_obs']

# Balance the Training Data using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Scale the Features using StandardScaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both training and test sets
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# Convert scaled NumPy arrays back to Pandas DataFrames to preserve column names and labels
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train_resampled.index)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)

# Train a Random Forest binary classification model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train_resampled)

# Evaluate the model
y_pred = model.predict(X_test_scaled)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Additional Metrics for Random Forest
f1_rf = f1_score(y_test, y_pred, average='binary', pos_label=9)  # For binary classification
mcc_rf = matthews_corrcoef(y_test, y_pred)
kappa_rf = cohen_kappa_score(y_test, y_pred)

print("Random Forest Evaluation Metrics:")
print(f"F1 Score: {f1_rf:.2f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_rf:.2f}")
print(f"Kappa Statistic: {kappa_rf:.2f}")

# Train SVM model
svm_model = SVC(probability=True, random_state=42)
svm_model.fit(X_train_scaled, y_train_resampled)  # Train the SVM model using scaled data

# Evaluate the SVM model
y_pred_svm = svm_model.predict(X_test_scaled)  # Predict using test data for SVM
print("\nClassification Report for SVM:\n", classification_report(y_test, y_pred_svm))

# Additional Metrics for SVM
f1_svm = f1_score(y_test, y_pred_svm, average='binary', pos_label=9)
mcc_svm = matthews_corrcoef(y_test, y_pred_svm)
kappa_svm = cohen_kappa_score(y_test, y_pred_svm)

print("\nSVM Evaluation Metrics:")
print(f"F1 Score: {f1_svm:.2f}")
print(f"Matthews Correlation Coefficient (MCC): {mcc_svm:.2f}")
print(f"Kappa Statistic: {kappa_svm:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tropical seasonal forest', 'Tropical rain forest'], yticklabels=['Tropical seasonal forest', 'Tropical rain forest'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Confusion Matrix for SVM
cm_svm = confusion_matrix(y_test, y_pred_svm)
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tropical seasonal forest', 'Tropical rain forest'], yticklabels=['Tropical seasonal forest', 'Tropical rain forest'])
plt.title('Confusion Matrix (SVM)')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Cumulative Gain or Lift Chart
from scikitplot.metrics import plot_cumulative_gain
plot_cumulative_gain(y_test, model.predict_proba(X_test_scaled))
plt.title('Cumulative Gains Curve')
plt.show()

# Get predicted probabilities for the positive class (1) for each model
y_prob_rf = model.predict_proba(X_test_scaled)[:, 1]  # Random Forest probabilities
y_prob_svm = svm_model.decision_function(X_test_scaled)  # SVM decision function (for linear SVM)

# Define the hyperparameter grid
param_dist = {
    'n_estimators': [50, 100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

# Implement RandomizedSearchCV
random_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=20,  # Number of parameter settings sampled
    cv=5,  # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1  # Use all available cores
)

# Fit the RandomizedSearchCV model
random_search.fit(X_train_scaled, y_train_resampled)

# Best parameters from RandomizedSearchCV
best_random_model = random_search.best_estimator_
print(f"Best parameters (RandomizedSearchCV): {random_search.best_params_}")

# Evaluate the best random search model
y_pred_best_random = best_random_model.predict(X_test_scaled)
print("\nClassification Report for Best Random Model:\n", classification_report(y_test, y_pred_best_random))

# Compute feature importances from the best random search model
importances = best_random_model.feature_importances_

# Get the feature names
feature_names = X_train_scaled.columns

# Create a DataFrame for the feature importances
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})

# Sort the DataFrame by importance
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the top 10 most important features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(10))
plt.title('Top 10 Feature Importances')
plt.show()

# Train an SVM with a linear kernel
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train_resampled)

# Get feature importances (coefficients) from SVM
svm_coefficients = svm_linear.coef_.flatten()
# Create a DataFrame for Random Forest feature importances
feature_importance_rf = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Importance': model.feature_importances_
})

# Create a DataFrame for SVM coefficients
feature_importance_svm = pd.DataFrame({
    'Feature': X_train_scaled.columns,
    'Coefficient': svm_coefficients
})

# Sort the features by importance for better visualization (top 10)
feature_importance_rf = feature_importance_rf.sort_values(by='Importance', ascending=False).head(10)
feature_importance_svm = feature_importance_svm.reindex(feature_importance_rf['Feature'])

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(16, 6))

# Random Forest Feature Importances
sns.barplot(x='Importance', y='Feature', data=feature_importance_rf, ax=ax[0], hue='Feature', palette="viridis", legend=False)
ax[0].set_title('Top 10 Feature Importances (Random Forest)')

# SVM Coefficients
sns.barplot(x='Coefficient', y='Feature', data=feature_importance_svm, ax=ax[1], hue='Feature', palette="plasma", legend=False)
ax[1].set_title('Top 10 Feature Coefficients (SVM with Linear Kernel)')

plt.tight_layout()
plt.show()

# Plotting the final random forest model confusion matrix:
cm = confusion_matrix(y_test, y_pred_best_random)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Tropical seasonal forest', 'Tropical rain forest'], yticklabels=['Tropical seasonal forest', 'Tropical rain forest'])
plt.title('Confusion Matrix')
plt.ylabel('Actual Class')
plt.xlabel('Predicted Class')
plt.show()

# Plotting misclassified points from the best random forest model
# Get indices of misclassified points
misclassified_indices = np.where(y_test != y_pred_best_random)[0]

# Create a DataFrame with misclassified points
misclassified_points = X_test.iloc[misclassified_indices].copy()
misclassified_points['Actual'] = y_test.iloc[misclassified_indices].values
misclassified_points['Predicted'] = y_pred_best_random[misclassified_indices]

try:
    # Load the world map
    world = gpd.read_file(r'C:\Users\asus\Desktop\Lund\Computational Modelling\Machine Learning Forest\110m_cultural')
    
    # Create a figure and axis for plotting
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    # Plot country boundaries
    world.boundary.plot(ax=ax, edgecolor='black', linewidth=0.8)

    # Plot misclassified points, color-coded by their actual class
    misclassified_gdf = gpd.GeoDataFrame(
        misclassified_points,
        geometry=gpd.points_from_xy(misclassified_points['Lon'], misclassified_points['Lat']),
        crs='EPSG:4326'
    )

    # Plot misclassified points on the same axis
    misclassified_gdf.plot(ax=ax, column='Actual', cmap='viridis', markersize=10)

    # Set plot title and labels
    ax.set_title('Misclassified Points from Best Random Forest Model')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_xlim([-85, -30])  # Longitude range for the Americas
    ax.set_ylim([-60, 15])   # Latitude range for the Americas

    plt.tight_layout()
    plt.show()

except fiona.errors.DriverError as e:
    print("Error loading boundary file:", e)
    print("Please verify the path or download the required geographic data.")

# Save the cleaned dataset for future use
import os

# Create the 'out' directory if it does not exist
os.makedirs('out', exist_ok=True)

data_cleaned.to_csv(r'out/data_cleaned.csv', index=False)
print("\nData cleaning and model training complete!")
