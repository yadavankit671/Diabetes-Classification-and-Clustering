# %% [markdown]
# # Diabetes Classification and Clustering Analysis
# 
# This notebook explores a diabetes dataset using both supervised classification and unsupervised clustering. The workflow includes data preparation, exploratory analysis, feature assessment, model training, clustering, and a comparison of the two approaches.
# 
# ## 1. Libraries and dependencies
# 
# The analysis starts by importing the libraries needed for data handling, visualization, preprocessing, modeling, and evaluation.
# 

# %%
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans, HDBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay, adjusted_rand_score, normalized_mutual_info_score

# %% [markdown]
# ## 2. Data ingestion
# 
# Load the dataset and inspect the first rows, schema, and sample records to confirm the structure before any cleaning or modeling.
# 

# %%
df = pd.read_csv("Dataset of Diabetes .csv")
df.head()

# %%
df.info()

# %% [markdown]
# ## 3. Data cleaning and preprocessing
# 
# Remove unneeded columns, normalize labels, handle duplicates, and inspect invalid values so the data is ready for modeling.
# 

# %%
df = df.drop(columns=['ID', 'No_Pation'],axis = 1)

# %%
df.head(5)

# %%
df.describe()

# %%
for col in df.columns:
    print(f"\n{col}:")
    print(df[col].unique())

# %%
df['Gender'] = df['Gender'].astype(str).str.upper().map({'M': 0, 'F': 1}).fillna(df['Gender'])
df['CLASS'] = df['CLASS'].astype(str).str.strip()

print(df['Gender'].value_counts())
print(df['CLASS'].value_counts())

# %%
print("No. of duplicate rows:", df.duplicated().sum())

# %%
df= df.drop_duplicates()
print("No. of duplicate rows:", df.duplicated().sum())

# %%
numeric_cols = df.select_dtypes(include=['number']).columns

print(df[numeric_cols].describe().T)

# %%
invalid_cols = []
for col in numeric_cols:
    if col == 'Gender' : 
        continue
    if (df[col] <= 0).any():
        invalid_cols.append(col)
        print(f"\nInvalid values in {col}:")
        print(df.loc[df[col]<=0])

# %%
for col in invalid_cols:
    df = df[df[col] > 0]

# %% [markdown]
# ## 4. Exploratory data analysis
# 
# Explore class balance, distributions, relationships, and correlation patterns to understand the healthcare features and the signals driving the target class.
# 

# %%
class_percentage = df['CLASS'].value_counts(normalize=True) * 100
print(f"\nPercentage of each class:\n{class_percentage}")

# %%
classes = sorted(df['CLASS'].unique())
for col in numeric_cols:
    data = [df[df['CLASS'] == cls][col] for cls in classes]
    plt.figure(figsize=(8, 5))
    plt.boxplot(data, tick_labels=classes) 
    
    plt.title(f"{col} by Diabetes Class")
    plt.xlabel("CLASS")
    plt.ylabel(col)
    plt.show()

# %%
for col in numeric_cols:
    plt.figure(figsize=(8, 4))
    plt.hist(df[col], bins=30)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")

    plt.show()

# %%
plt.figure(figsize=(12, 8))
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Metabolic Markers")
plt.show()

# %% [markdown]
# ## 5. Model building and training
# 
# Create the target and feature variables, split the dataset, standardize inputs, and train several classifiers along with an artificial neural network for comparison.
# 

# %%
Y = df['CLASS']
X = df.drop(columns=['CLASS'], axis=1)

# %%
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)

# %%
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# %%
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)

# %%
nn_model = Sequential([
    Input(shape=(x_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(24, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dropout(0.3),
    Dense(len(encoder.classes_), activation='softmax')
])

# %%
nn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# %%
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

# %%
class_labels = np.unique(y_train_encoded)
weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train_encoded)
class_weights = dict(zip(class_labels, weights))

# %%
print("Class weights:", class_weights)

# %%
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight='balanced',
        random_state=42
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight='balanced',
        random_state=42
    ),
    "Support Vector Machine": SVC(
        class_weight=class_weights,
        random_state=42
    ),
    "ANN": nn_model
}

# %% [markdown]
# ## 6. Classification results
# 
# Evaluate the trained classifiers using accuracy and macro-average metrics, and review confusion matrices and performance comparisons to identify the strongest model.
# 

# %%
results = []
trained_models = {}

for name, model in models.items():
    if name in ["Logistic Regression", "Support Vector Machine"]:
        model.fit(x_train_scaled, y_train_encoded)
        y_pred = model.predict(x_test_scaled)
    
    elif name == "ANN":
        model.fit(
            x_train_scaled, 
            y_train_encoded,
            epochs=100, 
            batch_size=32, 
            validation_split=0.2, 
            callbacks=[early_stopping],
            class_weight=class_weights,
            verbose=0
        )
        y_pred_probs = model.predict(x_test_scaled, verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
    else:
        model.fit(x_train, y_train_encoded)
        y_pred = model.predict(x_test)
    
    trained_models[name] = model
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    classification_rep = classification_report(
        y_test_encoded, 
        y_pred, 
        target_names=encoder.classes_, 
        output_dict=True, 
        zero_division=0
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Classification Report": classification_rep
    })

# %%
results_df = pd.DataFrame(results)
results_df = results_df.sort_values(
    by="Accuracy", ascending=False
)

print("Model Performance")
print(
    results_df[["Model", "Accuracy"]].to_string(
        index=False,
        formatters={"Accuracy": "{:.2%}".format}
    )
)

for _, row in results_df.iterrows():
    print(f"\n{row['Model']} - Classification Report")
    report_df = pd.DataFrame(row["Classification Report"]).T
    print(report_df.to_string(float_format="{:.3f}".format))

# %%
results_df['Precision'] = results_df['Classification Report'].apply(lambda x: x['macro avg']['precision'])
results_df['Recall'] = results_df['Classification Report'].apply(lambda x: x['macro avg']['recall'])
results_df['F1-Score'] = results_df['Classification Report'].apply(lambda x: x['macro avg']['f1-score'])

plt.figure(figsize=(10, 6))

x_coord = np.arange(len(results_df))
width = 0.2

plt.bar(
    x_coord - 1.5 * width,
    results_df['Accuracy'],
    width,
    label='Accuracy'
)

plt.bar(
    x_coord - 0.5 * width,
    results_df['Precision'],
    width,
    label='Precision'
)

plt.bar(
    x_coord + 0.5 * width,
    results_df['Recall'],
    width,
    label='Recall'
)

plt.bar(
    x_coord + 1.5 * width,
    results_df['F1-Score'],
    width,
    label='F1 Score'
)

plt.xticks(
    x_coord,
    results_df['Model'],
    rotation=45
)

plt.ylabel("Score")
plt.title("Classification Model Comparison (Macro Average)")

plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

plt.ylim(0, 1.05)
plt.tight_layout()

plt.show()

# %%
predictions = {}

for name, model in trained_models.items():
    if name in ["Logistic Regression", "Support Vector Machine"]:
        y_pred_int = model.predict(x_test_scaled)
        
    elif name == "ANN":
        y_pred_probs = model.predict(x_test_scaled, verbose=0)
        y_pred_int = np.argmax(y_pred_probs, axis=1)
        
    else:
        y_pred_int = model.predict(x_test)
    
    y_pred = encoder.inverse_transform(y_pred_int)
    
    predictions[name] = y_pred

# %%
for name, y_pred in predictions.items():
    print(f"\n{name} Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

# %%
for name, y_pred in predictions.items():
    
    conf_mat = confusion_matrix(y_test, y_pred, labels=sorted(Y.unique()))
    conf_mat_disp = ConfusionMatrixDisplay(confusion_matrix=conf_mat, display_labels=sorted(Y.unique()))
    
    conf_mat_disp.plot() 
    plt.title(f"Confusion Matrix - {name}")
    plt.show()

# %%
rf_model = trained_models["Random Forest"]

feature_importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values(ascending=False)

print(feature_importance)

# %%
plt.figure(figsize=(10, 6))

feature_importance.sort_values().plot(
    kind='barh'
)

plt.title("Random Forest Feature Importance")
plt.xlabel("Importance")

plt.show()

# %% [markdown]
# ## 7. Clustering
# 
# Reduce dimensionality with PCA and apply unsupervised learning algorithms to identify latent groups in the diabetes data without using the class labels.
# 

# %%
x_scaled_full = scaler.fit_transform(X) 
y_encoded_full = encoder.fit_transform(Y)

# %%
pca_full = PCA()
pca_full.fit(x_scaled_full)

cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
variance_threshold = 0.90
best_n_components = np.argmax(cumulative_variance >= variance_threshold) + 1

print(f"Optimal number of components for {variance_threshold*100}% variance: {best_n_components}")


pca_final = PCA(n_components=best_n_components)
X_pca = pca_final.fit_transform(x_scaled_full)

print(f"Original Data Shape: {x_scaled_full.shape}")
print(f"Reduced PCA Data Shape: {X_pca.shape}")
print(f"Exact Variance Retained: {np.sum(pca_final.explained_variance_ratio_) * 100:.2f}%")

# %%
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans_clusters = kmeans.fit_predict(X_pca)

agglo = AgglomerativeClustering(n_clusters=3)
agglo_clusters = agglo.fit_predict(X_pca)

hdb = HDBSCAN(min_cluster_size=15)
hdb_clusters = hdb.fit_predict(X_pca)


# %% [markdown]
# ## 8. Clustering results
# 
# Measure cluster quality using Adjusted Rand Index and Normalized Mutual Information to compare how well each clustering algorithm aligns with the known class labels.
# 

# %%
clustering_results = [
    {
        "Algorithm": "K-Means",
        "ARI Score": adjusted_rand_score(y_encoded_full, kmeans_clusters),
        "NMI Score": normalized_mutual_info_score(y_encoded_full, kmeans_clusters)
    },
    {
        "Algorithm": "Agglomerative",
        "ARI Score": adjusted_rand_score(y_encoded_full, agglo_clusters),
        "NMI Score": normalized_mutual_info_score(y_encoded_full, agglo_clusters)
    },
    {
        "Algorithm": "HDBSCAN",
        "ARI Score": adjusted_rand_score(y_encoded_full, hdb_clusters),
        "NMI Score": normalized_mutual_info_score(y_encoded_full, hdb_clusters)
    }
]

cluster_df = pd.DataFrame(clustering_results)
print("Clustering Algorithm Performance")
print(cluster_df.to_string(index=False, float_format="{:.3f}".format))

# %% [markdown]
# ## 9. Classification vs clustering comparison
# 
# Compare supervised and unsupervised approaches side by side to understand their differences in performance and how clustering structure relates to the labeled diabetes classes.
# 

# %%
# Classification vs Clustering Comparison

classification_summary = results_df[
    ["Model", "Accuracy", "F1-Score"]
].copy()

clustering_summary = cluster_df[
    ["Algorithm", "ARI Score", "NMI Score"]
].copy()

print("Classification Performance")
display(
    classification_summary.style.format({
        "Accuracy": "{:.2%}",
        "F1-Score": "{:.3f}"
    })
)

print("Clustering Performance")
display(
    clustering_summary.style.format({
        "ARI Score": "{:.3f}",
        "NMI Score": "{:.3f}"
    })
)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

classification_summary.set_index("Model")[["Accuracy", "F1-Score"]].plot(
    kind="bar",
    ax=axes[0],
    ylim=(0, 1.05),
    color=["steelblue", "darkorange"]
)
axes[0].set_title("Classification Performance")
axes[0].set_ylabel("Score")
axes[0].tick_params(axis="x", rotation=30)
axes[0].legend(loc="lower right")

clustering_summary.set_index("Algorithm")[["ARI Score", "NMI Score"]].plot(
    kind="bar",
    ax=axes[1],
    ylim=(0, 1.05),
    color=["seagreen", "mediumpurple"]
)
axes[1].set_title("Clustering Performance")
axes[1].set_ylabel("Score")
axes[1].tick_params(axis="x", rotation=30)
axes[1].legend(loc="lower right")

plt.suptitle("Classification vs Clustering Comparison")
plt.tight_layout()
plt.show()

best_classifier = classification_summary.loc[
    classification_summary["Accuracy"].idxmax()
]
best_cluster = clustering_summary.loc[
    clustering_summary["ARI Score"].idxmax()
]

print(f"Best classifier: {best_classifier['Model']} "
      f"({best_classifier['Accuracy']:.2%} accuracy)")
print(f"Best clustering algorithm by ARI: {best_cluster['Algorithm']} "
      f"({best_cluster['ARI Score']:.3f} ARI)")


# %% [markdown]
# ## 10. Error analysis
# 
# Inspect incorrect predictions to identify which patient patterns are hardest for the classifier and which features contribute most to misclassification.
# 

# %%
best_model_name = "Random Forest"
rf_pred = encoder.inverse_transform(trained_models[best_model_name].predict(x_test))

error_df = x_test.copy()
error_df['Actual'] = y_test
error_df['Predicted'] = rf_pred
error_df['Correct'] = error_df['Actual'] == error_df['Predicted']

p_confusions = error_df[
    (~error_df['Correct']) & 
    ((error_df['Actual'] == 'P') | (error_df['Predicted'] == 'P'))
]

print(f"\nMisclassifications involving the 'P' (Predict-diabetic) class: {len(p_confusions)} cases")
print(p_confusions[['HbA1c', 'BMI', 'Actual', 'Predicted']].head(10))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.scatterplot(
    x='Urea', y='Cr', 
    hue='Correct', 
    palette={True: 'gray', False: 'red'}, 
    style='Correct', markers={True: 'o', False: 'X'}, 
    data=error_df, ax=axes[0], alpha=0.7
)
axes[0].set_title("Urea vs Cr: Correct vs Misclassified")

sns.boxplot(x='Correct', y='Urea', data=error_df, ax=axes[1], palette='pastel', hue='Correct')
axes[1].set_title("Urea Distribution: Correct vs Incorrect Predictions")

plt.tight_layout()
plt.show()


