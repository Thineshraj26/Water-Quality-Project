#!/usr/bin/env python
# coding: utf-8

# ### Title: Smarter Water Quality Monitoring by Analyzing Relationships Between Water Quality Indicators and Predicting Potability Using Machine Learning in Environmental Engineering.
# 
# 
# This notebook contains four main sections addressing different research questions on the water potability dataset.

# ### Data Cleaning
# 
# Import Panda Library

# In[603]:


# Import libraries
import pandas as pd


# ### Load the Dataset

# In[606]:


# Load the dataset
df = pd.read_csv('water_potability.csv')

# Display the first few rows
df.head()


# ### Handling Missing Values 
# Rows with missing values were removed to clean the dataset

# In[609]:


# Drop missing values
df_clean = df.dropna()

# Check how many rows and columns are left
print(df_clean.shape)


# ### Research Question 1: What relationships exist among water quality indicators, and can clustering reveal meaningful patterns in their variation?
# 
# Objective: To investigate relationships between water quality indicators and use clustering techniques to identify patterns in their variations. 

# ### Import Necessary Libraries
# 
# 

# In[613]:


# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ### 1. Data Pre-processing/Cleaning
# 
# Goal: to prepare the dataset for unsupervised clustering by removing the target label and standardizing the feature values to ensure fair distance-based comparison.
# 
# The "Potability" column is first eliminated from the cleaned dataset in this code by using drop(). This is due to the fact that we are performing clustering, or unsupervised learning, and we do not want the model to use the known labels. StandardScaler() is then used to standardise all the remaining features, transforming the data so that each feature has a standard deviation of one and a mean of zero. Clustering techniques such as KMeans depend on distance computations, and if features on different scales are not standardised, the results may be skewed. This is why this phase is crucial.

# In[616]:


# Remove the target variable, Potability since clustering is unsupervised 
df_features = df_clean.drop("Potability", axis=1)

# Standardize the features to have mean 0 and standard deviation 1
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_features)


# ### 2. Data Analysis/ Statistical Study
# 
# Goal: To analyze the relationships between different water quality features by visualizing their correlations, helping us understand which features are strongly or weakly related.
# 
# Using Seaborn, this code generates a correlation heatmap that illustrates the relationship between each pair of water quality indicators. After setting the figure size, the correlation matrix, whose values range from -1 to 1, is displayed using sns.heatmap(). Strong positive correlations are shown by values near 1, strong negative correlations are indicated by values near -1, and no correlation is indicated by a value of 0. A distinct colour gradient is provided by cmap="coolwarm" and the numeric correlation values are shown inside the heatmap when the annot=True option is used. Prior to modelling, this visualisation provides insights into the structure of the data and aids in identifying redundant or related elements.

# In[619]:


# Plot a heatmap to see correlations between different water quality indicators
plt.figure(figsize=(10, 6))
sns.heatmap(df_features.corr(), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Water Quality Indicators")
plt.tight_layout()
plt.show()


# ### Key Insights
# 
# The correlation heatmap shows that, with correlation coefficients near zero, the majority of water quality indicators have a weak link with one another. This suggests that the features function mainly independently, which is advantageous for unsupervised clustering since every variable may provide distinct information. There are no significant positive or negative correlations found between any two features, indicating that multicollinearity is not an issue in this dataset. Consequently, it is possible to keep all features for modelling without redundancy. Overall, the heatmap facilitates the identification of unique water profiles by utilising all relevant indicators in the clustering process.

# ### 3. Modelling
# 
# Goal: to identify natural groupings or clusters within the water quality data using the KMeans clustering algorithm and to determine the optimal number of clusters using the Elbow Method and Silhouette Score.
# 
# This code groups similar water samples according to their quality indicators by applying the KMeans clustering algorithm to the standardised dataset (df_scaled). First, a range of cluster values (k) between 2 and 9 is defined, and KMeans is performed for each value. It computes the silhouette score, which indicates how well-separated the clusters are, and the inertia, which indicates how compact the clusters are, for each k and stores the results in lists. The ideal number of clusters is then visually ascertained by searching for the "elbow point" which is the point at which inertia begins to level off using the Elbow Method plot.

# In[623]:


inertia = []         # To store inertia values for elbow method
silhouette = []      # To store silhouette scores for cluster quality
k_range = range(2, 10)

# Try clustering for different values of k to find the best one
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(df_scaled, kmeans.labels_))

# Plot Elbow Method to help choose the optimal number of clusters
plt.figure()
plt.plot(k_range, inertia, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.tight_layout()
plt.show()

# Final KMeans clustering using the chosen number of clusters which is k=5
kmeans = KMeans(n_clusters=5, random_state=42)
df_features["Cluster"] = kmeans.fit_predict(df_scaled)


# ### Key Insights
# 
# Since additional clusters typically result in tighter groupings, the Elbow Method plot exhibits a consistent decrease in inertia as the number of clusters rises from two to nine. However, around k = 5, the rate of reduction noticeably slows down, creating a faint "elbow" point. This implies that selecting five clusters strikes a fair balance between eliminating needless complexity and lowering variation within clusters. Beyond this point, the benefits of increased compactness decrease as additional clusters are added. Consequently, it seems that k = 5 is a sensible and effective option for segmenting the water quality data into significant groups.

# ### 4. Evaluation
# 
# Goal: To evaluate the average feature profiles of each cluster and visualize cluster separation in reduced dimensions in order to assess and explain the clustering results. 
# 
# The code simplifies the visualisation of clusters in two dimensions by reducing the scaled dataset (df_scaled) to two primary components using Principal Component Analysis (PCA). PCA helps identify patterns or groupings by identifying the directions of greatest variance in the data. Each data point in this 2D space is then shown in a scatter plot, with colours indicating which cluster each point belongs to (df_features["Cluster"]). This helps in evaluating the clustering's effectiveness. Finally, to provide information about what makes each cluster distinct, the code computes and outputs the average values of each original feature for each cluster.

# In[627]:


# Apply PCA to reduce the scaled features to 2 dimensions for easier visualization
pca = PCA(n_components=2)
components = pca.fit_transform(df_scaled)

# Create a scatter plot of the clusters in the 2D PCA space
plt.figure(figsize=(7, 5))
sns.scatterplot(x=components[:, 0], y=components[:, 1], hue=df_features["Cluster"], palette="Set2")
plt.title("PCA Projection of Clusters")
plt.xlabel("Principal Component 1")  
plt.ylabel("Principal Component 2")  
plt.tight_layout()
plt.show()

# Calculate the average values of each feature in every cluster
cluster_means = df_features.groupby("Cluster").mean().round(2)

# Print the average feature values for each cluster
print(cluster_means)


# ### Key Insights
# 
# Five groups with differing levels of separation are seen in the PCA projection, suggesting that the clustering algorithm has found unique patterns in the water quality data. Although there is considerable overlap, other clusters, such Clusters 2 and 3, have more distinct borders. Each cluster's distinct qualities are highlighted through an analysis of its average feature values. Cluster 1 has high solids and conductivity, indicating mineral-heavy water, while Cluster 0 shows neutral pH and balanced chemical levels. The highest conductivity, sulphate, and trihalomethanes are found in Cluster 2, which may be a sign of chemical pollution. Cluster 3 indicates more murky and possibly less clean water because it has a lower pH and higher turbidity. The largest concentrations of organic carbon, solids, and trihalomethanes are seen in Cluster 4, which may indicate organic or industrial pollution. In summary, the PCA visualization and clustering successfully distinguish water samples according to their water quality indicators.

# ### Research Question 2: What is the overall distribution of key water quality indicators which strongly influence potability, and how do these indicators differ between potable and non-potable samples?
# 
# Objective: Summarize key statistics (mean, median, standard deviation, range) of water quality features using basic descriptive analysis and visual comparison.

# ### Import Necessary Libraries

# In[632]:


import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split


# ## 1. Data Pre-processing / Cleaning
# 
# The dataset was already cleaned at the top of the notebook by dropping rows with missing values using `df_clean = df.dropna()`. Therefore, no additional pre-processing was needed for this part. All analysis below uses the shared cleaned dataset `df_clean`.

# #### Selection of Key Water Quality Indicators
# 
# The **top 5 indicators (pH, Sulfate, Solids, Turbidity, Chloramines)** were selected based on domain knowledge and their importance in water quality assessments, as highlighted by **WHO (2017)** guidelines, rather than on statistical correlation.

# ### 2. Data Analysis/ Statistical Study
# 
# #### Summary Statistics of Key Water Quality Indicators
# The table shows mean, median, standard deviation, minimum, and maximum for key features grouped by Potability.

# In[637]:


# Compute grouped descriptive statistics
grouped_stats = df_clean.groupby("Potability")[["ph", "Sulfate", "Solids", "Turbidity", "Chloramines"]].agg(
    ['mean', 'median', 'std', 'min', 'max']
)

# Adjust display settings so full table shows
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

# Display the table
grouped_stats


# - **pH:** Potable water shows a slightly higher mean and median pH.
# - **Sulfate:** Potable water has a slightly lower Sulfate mean and median.
# - **Solids:** Similar means and medians across both classes; overlapping spread.
# - **Turbidity:** Minimal difference between potable and non-potable samples.
# - **Chloramines:** Comparable values between both groups.

# #### % difference in means

# In[641]:


# Percent difference in means between potable and non-potable
for col in ["ph", "Sulfate", "Solids", "Turbidity", "Chloramines"]:
    mean_potable = df_clean[df_clean["Potability"] == 1][col].mean()
    mean_nonpotable = df_clean[df_clean["Potability"] == 0][col].mean()
    diff = ((mean_potable - mean_nonpotable) / mean_nonpotable) * 100
    print(f"{col}: Potable mean is {diff:.2f}% {'higher' if diff>0 else 'lower'} than non-potable mean")


# #### Visual Comparison of Features by Potability
# 
# **Boxplots and density plots below compare distributions of selected features between potable and non-potable samples.**

# #### ph Boxplot

# In[645]:


plt.figure(figsize=(8, 5))
sns.boxplot(x="Potability", y="ph", data=df_clean)
plt.title("pH Distribution by Potability")
plt.xlabel("Potability (0 = Non-potable, 1 = Potable)")
plt.ylabel("pH")
plt.show()


# #### Sulfate Boxplot

# In[648]:


plt.figure(figsize=(8, 5))
sns.boxplot(x="Potability", y="Sulfate", data=df_clean)
plt.title("Sulfate Distribution by Potability")
plt.xlabel("Potability (0 = Non-potable, 1 = Potable)")
plt.ylabel("Sulfate (mg/L)")
plt.show()


# #### Solids Boxplot

# In[651]:


plt.figure(figsize=(8, 5))
sns.boxplot(x="Potability", y="Solids", data=df_clean)
plt.title("Solids Distribution by Potability")
plt.xlabel("Potability (0 = Non-potable, 1 = Potable)")
plt.ylabel("Solids (ppm)")
plt.show()


# #### Turbidity Boxplot

# In[654]:


plt.figure(figsize=(8, 5))
sns.boxplot(x="Potability", y="Turbidity", data=df_clean)
plt.title("Turbidity Distribution by Potability")
plt.xlabel("Potability (0 = Non-potable, 1 = Potable)")
plt.ylabel("Turbidity (NTU)")
plt.show()


# #### Chloramines Boxplot

# In[657]:


plt.figure(figsize=(8, 5))
sns.boxplot(x="Potability", y="Chloramines", data=df_clean)
plt.title("Chloramines Distribution by Potability")
plt.xlabel("Potability (0 = Non-potable, 1 = Potable)")
plt.ylabel("Chloramines (ppm)")
plt.show()


# #### Key Insights
# 
# - Non-potable water samples generally show lower pH and higher Sulfate levels.
# - Solids, Turbidity, and Chloramines distributions overlap but show slight shifts between groups.
# - No single feature perfectly separates potable from non-potable water. The patterns are subtle and multi-factorial, suggesting that potability depends on a combination of indicators rather than any single one.

# 
# 
# Descriptive analysis shows that potable water tends to have higher pH and lower Sulfate levels compared to non-potable water. Solids, Turbidity, and Chloramines also show differences, though with overlapping distributions. Overall, potability likely depends on a combination of indicators rather than any single one.

# ### Research Question 3: Does Hardness increase with Organic Carbon by potability, and what are their individual thresholds that best differ between potable and non-potable water?
# 
# Objective: To examine the correlation between hardness and organic carbon by potability and identify the threshold values that best differentiate potable from non-potable water.

# ### Import necessary libraries

# In[663]:


# Import necessary libraries for data manipulation, visualization and ROC analysis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np


# ### 1. Data Pre-processing / Cleaning
# 
# Goal: To isolate and prepare relevant features for analysis or visualisation in order to investigate the correlation between Hardness and Organic Carbon levels and how they may influence Potability.
# 
# The code starts by importing the necessary libraries which are Seaborn and Matplotlib for data visualisation, and Pandas for data manipulation. Next, it selects only the 'Hardness', 'Organic_carbon', and 'Potability' columns from an existing cleaned dataset called df_clean to construct a new DataFrame called df_clean_hardness_organic_carbon. This phase aids in focusing the investigation just on these three factors, which are being investigated to determine how hardness and organic carbon relate to one another and how they might affect the potability of the water. The algorithm prepares the data for more focused analysis or visualisation.

# In[666]:


# Select only the relevant columns for this analysis from the cleaned dataset
df_clean_hardness_organic_carbon = df_clean[['Hardness', 'Organic_carbon', 'Potability']].copy()


# ### 2. Data Analysis / Statistical Study
# 
# Goal: To investigate the differences in the chemical characteristics of potable and non-potable water samples, and to find any trends or separations that could aid in the classification of water potability 
# 
# The relationship between organic carbon and hardness in water,differentiated by potability status, is visualised by the code. The first plot helps identify possible patterns or correlations between the two variables for potable and non-potable water separately by using sns.lmplot() to generate a scatterplot with linear regression lines. The distributions and pairwise relationships between the variables Hardness and Organic Carbon are shown in the second visualisation which is a pairplot, coloured by potability.
# 

# In[669]:


# Visualize the relationship between Hardness and Organic Carbon and differentiated by Potability using linear regression lines
sns.lmplot(data=df_clean_hardness_organic_carbon, x='Organic_carbon', y='Hardness', hue='Potability', aspect=1.5)
plt.subplots_adjust(top=0.9)
plt.suptitle("Hardness vs Organic Carbon Regression by Potability")
plt.show()

# Generate a pairplot to observe the distribution and relationship between Hardness and Organic Carbon, colored by Potability
sns.pairplot(df_clean_hardness_organic_carbon, hue='Potability', vars=['Hardness', 'Organic_carbon'])
plt.suptitle("Pairwise Plot by Potability", y=1.02)
plt.show()


# ### Key Insights:
# 
# Regardless of potability, the regression plot and pairplot show a weak linear association between organic carbon and hardness. There appears to be no relationship between the regression lines for potable and non-potable water, which are almost flat. Furthermore, the distributions of Hardness and Organic Carbon for potable (orange) and non-potable (blue) water in the pairplot significantly overlap, suggesting that these factors by themselves are insufficient to distinguish the two groups. This implies that organic carbon and hardness may not be very good predictors of water potability on their own and should probably be examined alongside with other characteristics for improved classification results.

# ### 3. Evaluation
# 
# Goal: Evaluates the predictive power of Hardness and Organic Carbon individually in determining water potability.
# 
# The code assesses how effectively Hardness and Organic Carbon by themselves can predict the potability of water using ROC analysis. Employing Youden's J statistic, it determines the optimal threshold for each feature and produces the corresponding AUC values to assess predictive performance. ROC curves are presented to illustrate the trade-off between true and false positive rates.

# In[672]:


# Import additional libraries for ROC analysis
from sklearn.metrics import roc_curve, roc_auc_score
import numpy as np

# Define the target labels
y = df_clean_hardness_organic_carbon['Potability']

# Define a function to find the optimal classification threshold using Youden's J statistic (sensitivity - 1 - specificity)
def find_best_threshold(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr

# Get predicted scores from Hardness values
hardness_scores = df_clean_hardness_organic_carbon['Hardness']
hardness_threshold, fpr_hardness, tpr_hardness = find_best_threshold(y, hardness_scores)
auc_hardness = roc_auc_score(y, hardness_scores)

# Get predicted scores from Organic Carbon values
carbon_scores = df_clean_hardness_organic_carbon['Organic_carbon']
carbon_threshold, fpr_carbon, tpr_carbon = find_best_threshold(y, carbon_scores)
auc_carbon = roc_auc_score(y, carbon_scores)

# Plot ROC curves for both Hardness and Organic Carbon and include their AUC and optimal threshold in the legend
plt.figure(figsize=(7, 5))
plt.plot(fpr_hardness, tpr_hardness, 
         label=f'Hardness (AUC = {auc_hardness:.3f}, Threshold = {hardness_threshold:.2f})')
plt.plot(fpr_carbon, tpr_carbon, 
         label=f'Organic Carbon (AUC = {auc_carbon:.3f}, Threshold = {carbon_threshold:.2f})',
         linestyle='--')
plt.plot([0, 1], [0, 1], '--', color='gray')  # Reference line for random guess 

plt.title('ROC Curves for Hardness and Organic Carbon')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# ### Key Insights: 
# 
# Hardness and organic carbon are both poor individual predictors of water potability, according to the results of the ROC curve. Hardness and Organic Carbon have AUC (Area Under the Curve) score of 0.505 and 0.492, respectively, which are both near 0.5, meaning that they perform about as well as random guessing. Furthermore, both elements' ROC curves are near the diagonal reference line, which supports their poor discriminatory power. This implies that other informative features or a combination of several variables would be required for an efficient classification, as neither Hardness nor Organic Carbon alone can consistently differentiate between potable and non-potable water.

# ### Research Question 4: Can we predict water potability based on water quality indicators using machine learning?
# 
# Objective: To develop and assess a machine learning model that reliably predict water potability based on water quality indicators for early detection and quality management.
# 

# ### CatBoost Package Installation

# In[676]:


# Make sure to install the CatBoost package before running this code
get_ipython().system('pip install catboost')


# ### Import necessary libraries
# 

# In[680]:


# Import Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
import matplotlib.pyplot as plt
import seaborn as sn


# ### 1. Data Pre-processing/Cleaning
# 
# Goal: Prepare the dataset for modeling by transforming raw data into a format that is suitable and effective for machine learning model training
# 
# The process starts by importing all the necessary libraries for data handling, visualization, and modeling. A clean copy of the data was created to safely perform feature transformations .After that, some new features are created to help the model learn better patterns from the data.Next, the features (X) and target (y) are defined, where X includes all input variables and y represents whether the water is potable or not. The data is then split into training and testing sets, keeping the same class ratio.To fix class imbalance, SMOTE is applied to the training data, which adds synthetic examples of the minority class. This helps the model learn from both classes more equally.

# In[682]:


# Create a separate copy for modeling to avoid modifying the original cleaned dataset
df_modeling = df_clean.copy()

# ---Feature Engineering---

# Measures how far the pH level is from neutral (7)
df_modeling['Distance_from_neutral'] = abs(df_modeling['ph'] - 7)

# Ratio of total solids to conductivity — indicates dissolved solid effectiveness
df_modeling['Solids_per_Conductivity'] = df_modeling['Solids'] / df_modeling['Conductivity']

# Log-transform solids to reduce skew and impact of outliers
df_modeling['log_Solids'] = np.log(df_modeling['Solids'] + 1)

# Log-transform sulfate
df_modeling['log_Sulfate'] = np.log(df_modeling['Sulfate'] + 1)

# Ratio of organic carbon to turbidity which may reflect cloudiness from organic material
df_modeling['OrganicCarbon_per_Turbidity'] = df_modeling['Organic_carbon'] / df_modeling['Turbidity']

# ---Define features and target---

# Separate input features (X) and target variable (y)
X = df_modeling.drop('Potability', axis=1)
y = df_modeling['Potability']

# ---Split the data and apply SMOTE algorithm for class balancing---

# Stratified split to maintain class proportions
X_training_set, X_testing_set, y_training_set, y_testing_set = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Apply SMOTE to balance class distribution in the training set
smote = SMOTE(random_state=42)
X_training_resampled, y_training_resampled = smote.fit_resample(X_training_set, y_training_set)


# ### 2. Initial Modelling with Hyperparameter Tuning
# 
# Goal: To determine which combination of hyperparameters for the CatBoostClassifier produces the best overall classification performance, measured by the F1 macro score.
# 
# 
# This code uses GridSearchCV to hyperparameter tune the CatBoostClassifier, determining the optimal set of parameters to maximise model performance. It establishes a search space for the three main hyperparameters which are depth, iterations, and learning_rate and uses 5-fold cross-validation to assess each combination, guaranteeing that the model is reliable and performs well when applied to new data. The F1 macro score, which serves as the basis for the evaluation, balances recall and precision across all classes, making it appropriate for datasets that are unbalanced. The grid search is parallelized using all CPU cores (n_jobs=-1) and offer real-time progress output (verbose=1). After training on the resampled dataset, the model with the highest performance is taken out and saved for further usage in data analysis and final modeling, while the ideal parameter values are printed for reference.

# In[686]:


# ---Hyperparameter Tuning with Grid Search---

# Initialize CatBoost classifier
cat_model = CatBoostClassifier(verbose=0, random_state=42)

# Define hyperparameter search space
param_grid = {
    'depth': [4, 6],             # Controls how deep each decision tree grows
    'iterations': [100, 200],    # Number of trees the model will build
    'learning_rate': [0.1, 0.01] # Step size for updating the model after each tree
}

# Perform grid search with 5-fold cross-validation
grid_search = GridSearchCV(
    estimator=cat_model,
    param_grid=param_grid,
    scoring='f1_macro',    # ensures balanced performance across both classes
    cv=5,                  # 5-fold cross-validation ensures robust evaluation
    n_jobs=-1,             # use all CPU cores for faster processing
    verbose=1              # shows progress of the search
)

# Fit model on resampled training data
grid_search.fit(X_training_resampled, y_training_resampled)

# Extract the best estimator
best_cat = grid_search.best_estimator_
print("Best Parameters:", grid_search.best_params_)


# ### Key Insights:
# 
# The model underwent Grid Search with 5-fold cross-validation, testing 8 different hyperparameter combinations, resulting in a total of 40 training runs. The best-performing parameters were found to be a tree depth of 6, 200 iterations, and a learning rate of 0.1. These settings provided the highest performance based on the evaluation metric used.

# ### 3. Data Analysis / Statistical Study
# 
# Goal: Understand which features are most important and assess their impact on the model's decision-making process
# 
# In the statistical study and data analysis stage, feature significance scores which quantify each feature's contribution to the model's predictions were extracted from the trained initial CatBoost model. CatBoost's internal feature importance scores, which are derived from how much each feature contributes to reducing error across splits in the trees. were used to rank the features, and the top five most informative features were chosen for the final modelling. A bar chart that visualised the trankings of the attributes was made to aid in interpretation and show which factors were most important in predicting the potability of the water.

# In[689]:


# ---Feature Importance and Selection---

# Get feature importances from the best CatBoost model
feature_names = X.columns
importances = best_cat.get_feature_importance()

# Create DataFrame to rank features by importance
feature_df = pd.DataFrame({'Feature': feature_names,'Importance': importances}).sort_values(by='Importance', ascending=False)

# Select top 5 most important features
top_features = feature_df['Feature'].head(5).tolist()
print("Top 5 Features:", top_features)

# ---Feature Importance Plot---

# Plot features based on their importance score
plt.figure(figsize=(8, 5))
sns.barplot(
    x=feature_df['Importance'].head(14),
    y=feature_df['Feature'].head(14),
    palette='viridis'
)
plt.title("Feature Importances (CatBoost)")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()


# ### Key Insights:
# 
# The most important factors in predicting the potability of water are highlighted in the feature importance chart produced by the CatBoost model. pH, Hardness, Chloramines, Sulphate, and Distance_from_Neutral are the top five characteristics found. The fact that pH gets the highest significance score among them suggests that the acidity or alkalinity of water is a significant factor in determining whether or not it is safe to drink. Significant contributions are also made by hardness and chloramines, indicating that the main markers of potability are disinfectant levels and mineral content. On the other hand, characteristics with lower significance scores such as turbidity, solids, and their log-transformed versions had less influence on the model's judgement. Prioritising which water quality indicators are most important for determining potability is made easier by these insights.

# ### 4. Final Modeling
# 
# Goal: Develop a strong and useful predictive model capable of accurately predicting whether a given water sample is potable.
# 
# A CatBoostClassifier, a powerful gradient boosting model that excels at managing both categorical features and unbalanced datasets, was initialised at the start of the initial modelling procedure. Through 5-fold cross-validation, different combinations of parameters, including depth, number of iterations, and learning rate, were tested in order to optimise GridSearchCV's performance. Only the top five most crucial features from the feature importance analysis were used to retrain the final model after the best performing parameter set was determined. This method maintained high predictive accuracy while guaranteeing a simpler and more efficient model.

# In[692]:


# ---Retrain Final CatBoost Model---

# Filter training and test sets to include only top features
X_training_top = X_training_set[top_features]
X_testing_top = X_testing_set[top_features]
X_top = X[top_features]

# Reapply SMOTE on reduced feature set
X_training_top_resampled, y_training_top_resampled = smote.fit_resample(X_training_top, y_training_set)

# Train CatBoost on selected top features using best parameters
final_catboost = CatBoostClassifier(**grid_search.best_params_, verbose=0, random_state=42)
final_catboost.fit(X_training_top_resampled, y_training_top_resampled)


# ### 5. Evaluation
# 
# Goal: Assess the predictive model's performance to ensure it not only works well during training but also generalizes reliably to unseen data.
# 
# 5-fold cross-validation was used in the evaluation process to compute F1 macro scores across several data splits, offering a reliable assessment of the model's overall performance. After training, the model was tested on the unseen test set, where predictions were made and assessed using several metrics The model's performance on each class was clearly visible through the generation of a classification report that included precision, recall, and F1-score. To highlight the differences between actual and predicted values and emphasise accurate and inaccurate classifications, a confusion matrix was also made. In order to visualise the trade-off between true positive and false positive rates, the ROC AUC score and curve were also utilised to assess the model's capacity to differentiate between potable and non-potable water.

# In[694]:


# ---Cross-Validation Evaluation---

# Apply SMOTE on full dataset with top features only
X_resampled_all, y_resampled_all = smote.fit_resample(X_top, y)

# Use StratifiedKFold to preserve class balance in each fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Calculate cross-validated F1 macro scores
cv_scores = cross_val_score(
    final_catboost, X_resampled_all, y_resampled_all,
    scoring='f1_macro',
    cv=cv
)
print("CV F1 Macro Scores:", cv_scores)
print("Mean CV F1 Macro Score:", np.mean(cv_scores).round(3))

# ---Final Model Evaluation---

# Predict labels and probabilities on the test set
y_predict = final_catboost.predict(X_testing_top)
y_probability = final_catboost.predict_proba(X_testing_top)[:, 1]

# Print classification metrics
print("\nCatBoost Classification Report:")
print(classification_report(y_testing_set, y_predict))

# Display confusion matrix
conf_matrix = confusion_matrix(y_testing_set, y_predict)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='YlGnBu')
plt.title("Confusion Matrix (CatBoost)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Compute and print ROC AUC score
auc = roc_auc_score(y_testing_set, y_probability)
print(f"ROC AUC Score: {auc:.3f}")

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_testing_set, y_probability)
plt.figure(figsize=(6, 4))
plt.plot(fpr, tpr, label=f"CatBoost (AUC = {auc:.3f})", color='purple')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.title("ROC Curve (CatBoost)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.show()


# ### Key Insights:
# 
# With a mean cross-validated F1 macro score of 0.705 and a ROC AUC score of 0.733, the CatBoost model exhibits moderate classification performance, meaning it can fairly differentiate between the two classes. The model's performance on Class 0 (not potable) is higher, with a precision of 0.73 and recall of 0.80, according to the classification report. However, on Class 1 (potable), it performs worse, with a precision of 0.65 and a noticeably lower recall of 0.57. The confusion matrix, which shows that 70 instances of Class 1 are incorrectly categorised as Class 0, further illustrates this imbalance and suggests that the model has trouble accurately identifying positive cases. 
