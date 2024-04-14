# Final_project_PredictiveModeling

1.	Introduction
2.	Methodology
  * Data Preparation
	* Exploratory Data Analysis (EDA)
  * Data Preprocessing
  * Feature Selection
3.	Model Selection
  * Data Splitting
  * Model Training and Evaluation
4.	Hyperparameter Tuning
5.	Prediction
6.	Conclusion
7.	References




# 1.	Introduction

The integration of machine learning in nutritional science can provide significant insights into the relationships between dietary patterns and health outcomes. The project aims to build a predictive model for a recipe website that classifies food items into three nutritional categories: "Nourishing", "Indulgent", and "Balanced". This classification is based on various nutritional parameters such as protein, fiber, saturated fat, sugar, and caloric content. The objective is to leverage this classification to recommend healthier food choices to users. 

# 2.	Methodology
## * Data Preparation
##### Loading Libraries:

1. **Pandas**: Used for its powerful data manipulation capabilities that simplify data cleaning, transformation, and analysis.
2. **NumPy**: Essential for handling numerical operations especially in data with complex calculations.
3. **Matplotlib and Seaborn**: These libraries are pivotal for data visualization, offering a wide range of plotting tools that help in making comprehensive graphs and charts.
4. **Scikit-learn**: This library is fundamental for implementing machine learning algorithms, providing tools for data splitting, model building, and model evaluation.

## * Data Loading: 
The given train dataset was loaded for the project. The dataset is a comprehensive collection of nutritional metrics and food categories, compiled into a CSV file format which allows for easy access and manipulation using data analysis tools. 

## * Data Cleaning:
The following steps were meticulously carried out:
* Type Conversion: Correcting data types, such as converting strings with units in numerical columns to proper numeric types.
* Duplicate Removal: Ensuring the uniqueness of data entries to maintain analytical integrity by removing duplicate rows.

2.2.	Exploratory Data Analysis (EDA)

Univariate Analysis: Initial analysis focused on single variables to understand their distribution and inherent properties. Histograms and box plots revealed a variety of distribution shapes across different nutrients, such as calories, proteins, and fats. Specific variables demonstrated skewness or kurtosis, indicating the potential need for data transformation to normalize these features.

Bivariate Analysis: Scatter plots and correlation matrices were utilized to explore pairwise relationships between variables. This analysis helped uncover significant positive or negative correlations, such as the relationship between total fat and calorie content, which was expectedly strong and positive. This section also highlighted potential multicollinearity issues, which could impact the performance of certain machine learning models.

Multivariate Analysis: Advanced techniques including heatmaps and pair plots facilitated the examination of interactions among multiple variables. These analyses provided insights into complex interdependencies, such as how different nutrients contribute collectively to health categories or dietary classifications. It also helped identify patterns or clusters within the data, which could be pivotal for segmentation or targeted nutritional advice.


Observations: 

	Distribution of Variables: Understanding the distribution of various nutritional metrics like calories, proteins, fats, and carbohydrates. This can reveal patterns like normal distribution, skewness, or bimodality which are essential for choosing appropriate data transformation or normalization techniques.

	Outlier Detection: The application of statistical methods to detect outliers revealed several extreme values in variables like cholesterol and fiber content. Decisions were made on whether to trim these outliers or cap them at certain thresholds to prevent model distortion.

	Missing Value Analysis: The dataset contained missing entries in several columns such as Cholesterol, Fiber, and Vitamin C. Strategies for handling these missing values were developed based on the pattern of their occurrence and their impact on the overall dataset. In some cases, imputation with the mean or median was used, while in others, deletion of missing data was considered based on the proportion of missingness and the relevance of the variable to the study.

2.3.	Data Preprocessing

Data preprocessing is a critical phase in any machine learning project, involving the transformation and manipulation of raw data to prepare it for modeling. This section outlines the various steps undertaken to preprocess the dataset before model training.

Handling Missing Values:
Missing data can adversely affect model performance and interpretation. The following strategies were employed to handle missing values:

	Identification of Missing Values: Initially, missing values were identified across all columns of the dataset using Pandas' isnull() function, which returns a Boolean mask indicating the presence of missing values.

	Imputation Techniques: Missing values were imputed using appropriate techniques based on the nature of the data:
•	For numerical features, missing values were imputed with the mean or median of the respective column to preserve the distribution of the data and minimize bias.
•	For categorical features, missing values were imputed with the mode or a designated "unknown" category, ensuring the preservation of categorical structure.

Handling Outliers:
Outliers, or extreme values, can significantly skew statistical analyses and model predictions. The following steps were taken to address outliers:

	Outlier Detection: Outliers were identified using statistical methods such as the interquartile range (IQR) or Z-score.

	Treatment of Outliers: Outliers were treated through various methods, including:
•	Trimming: Extreme values lying beyond a certain threshold (e.g., 1.5 times the IQR) were trimmed or capped to prevent them from unduly influencing the analysis.
•	Outliers were replaced with values corresponding to the nearest non-outlying data point to mitigate their impact while preserving the overall distribution.

Feature Engineering:
Feature engineering involves creating new features or transforming existing ones to enhance model performance and interpretability. The following feature engineering techniques were applied:
	Creation of Derived Features: New features were generated to capture additional information or simplify complex relationships within the data. For example:
•	Calorie Density: A new feature representing the amount of calories per unit weight or volume of food, calculated by dividing total calories by total weight or volume.
•	Total Nutrient Content: A composite feature aggregating the nutritional content of multiple nutrients, calculated by summing the values of individual nutrients.

	Transformation of Skewed Features: Skewed features, identified through exploratory data analysis, were transformed using appropriate techniques such as logarithmic transformation to improve symmetry and normalize the distribution.

Feature Encoding:
Categorical variables require encoding into numerical format to facilitate model training. The following encoding techniques were employed:

	One-Hot Encoding: Categorical variables with multiple categories were encoded using one-hot encoding, creating binary dummy variables for each category. This ensures that each category is represented as a separate feature, avoiding ordinality assumptions.

	Label Encoding: For categorical variables with a natural order (e.g., ordinal variables), label encoding was applied to convert categories into ordinal integers. Care was taken to ensure that the encoding preserves the inherent order of the categories.

Data Scaling:
Feature scaling is essential for many machine learning algorithms to ensure that all features contribute equally to model training. The following scaling technique was applied:

	Standardization: Numeric features were standardized to have a mean of 0 and a standard deviation of 1 using Scikit-learn's StandardScaler. This transformation preserves the shape of the distribution while scaling the values to a comparable range, preventing features with larger magnitudes from dominating the mode.

2.4.	 Feature Selection

Feature selection is a crucial step in machine learning model development, as it involves identifying the most relevant features that contribute to the predictive power of the model while reducing dimensionality and computational complexity. This section outlines the comprehensive feature selection process undertaken in this project.

Variance Thresholding:
Constant features, i.e., those with zero variance or negligible variability, provide little to no discriminatory power and can be safely removed from the dataset. The following steps were taken to perform variance thresholding:

	Identification of Constant Features: Features with near-zero variance were identified using Scikit-learn's VarianceThreshold method, which allows for the removal of features whose variance does not exceed a specified threshold.

	Threshold Selection: A threshold of zero was chosen to identify and remove constant features, ensuring that only features with non-zero variance contribute to model training and prediction.

Feature Selection Techniques:
Several feature selection methods were explored to identify the most informative features for model training. The following techniques were employed:

	SelectKBest with ANOVA F-value Scoring: SelectKBest is a univariate feature selection method that evaluates the significance of each feature individually with respect to the target variable. ANOVA F-value scoring measures the degree of linear dependency between each feature and the target variable, with higher F-values indicating greater relevance. The following steps were taken:
•	Features were ranked based on their ANOVA F-values using Scikit-learn's f_classif function, which computes the F-value for classification tasks.
•	The top k features with the highest F-values were selected for inclusion in the final feature set, where k represents the desired number of features.

3.	Model Selection
Model selection is a crucial step in machine learning projects, as it involves choosing the most appropriate algorithm(s) for the task at hand and evaluating their performance against predefined metrics. This section provides a comprehensive overview of the model selection process undertaken in this project.

3.1.	Data Splitting:
Before model selection, the dataset was split into training and testing sets to facilitate unbiased evaluation of model performance. The following steps were taken:

	Partitioning of Data: The dataset was randomly partitioned into training and testing sets using Scikit-learn's train_test_split function. A common split ratio of 80% training data and 20% testing data was employed to ensure an adequate amount of data for model training while reserving sufficient data for evaluation.
3.2.	Model Training and Evaluation:
Multiple machine learning algorithms were considered for model training, each offering unique strengths and weaknesses. The following models were trained and evaluated:
1)	Logistic Regression:
	Model Overview: Logistic regression is a linear classification algorithm commonly used for binary classification tasks.
	Training and Prediction: The logistic regression model was trained on the training data using Scikit-learn's LogisticRegression class and subsequently used to predict the target variable on the testing data.
	Evaluation Metrics: Model performance was assessed using metrics such as accuracy, precision, recall, and F1-score, which provide insights into the model's ability to correctly classify instances and balance between true positive and false positive rates.

	Accuracy: 0.808
	Precision: 0.78 (Macro Avg), 0.80 (Weighted Avg)
	Recall: 0.77 (Macro Avg), 0.81 (Weighted Avg)
	F1 Score: 0.77 (Macro Avg), 0.81 (Weighted Avg)
	Cross-Validation Results:
	ROC AUC Score: 0.93

2)	Decision Tree:
	Model Overview: Decision trees are non-linear models that recursively partition the feature space based on the most informative features.
	Training and Prediction: The decision tree classifier was trained on the training data using Scikit-learn's DecisionTreeClassifier class and used to predict the target variable on the testing data.
	Evaluation Metrics: Model performance was evaluated using similar metrics as logistic regression, with additional consideration given to the tree's interpretability and ability to capture complex decision boundaries.
	Accuracy: 0.984
	Precision: 0.99 (Macro Avg), 0.98 (Weighted Avg)
	Recall: 0.99 (Macro Avg), 0.98 (Weighted Avg)
	F1 Score: 0.99 (Macro Avg), 0.98 (Weighted Avg)
	Cross-Validation Results:
	ROC AUC Score: 0.99

3)	Random Forest Classifier:
	Model Overview: Random forests are ensemble learning methods that aggregate multiple decision trees to improve predictive performance and robustness.
	Training and Prediction: The random forest classifier was trained on the training data using Scikit-learn's RandomForestClassifier class and used to predict the target variable on the testing data.
	Evaluation Metrics: Similar evaluation metrics as logistic regression and decision trees were used to assess model performance, with a focus on the ensemble's ability to reduce overfitting and variance.
	Accuracy: 0.990
	Precision: 0.98 (Macro Avg), 0.99 (Weighted Avg)
	Recall: 0.99 (Macro Avg), 0.99 (Weighted Avg)
	F1 Score: 0.99 (Macro Avg), 0.99 (Weighted Avg)
	Cross-Validation Results:
	ROC AUC Score: 1.00

4)	Gradient Boosting Classifier:
	Model Overview: Gradient boosting is a sequential ensemble learning technique that builds multiple weak learners sequentially to correct the errors of previous models.
	Training and Prediction: The gradient boosting classifier was trained on the training data using Scikit-learn's GradientBoostingClassifier class and used to predict the target variable on the testing data.
	Evaluation Metrics: Model performance was assessed using the same set of evaluation metrics.
	Accuracy: 1.00
	Precision: 1.00 (Macro Avg), 1.00 (Weighted Avg)
	Recall: 1.00 (Macro Avg), 1.00 (Weighted Avg)
	F1 Score: 1.00 (Macro Avg), 1.00 (Weighted Avg)
4.	Hyperparameter Tuning

Given the exemplary performance of the Gradient Boosting Classifier, further hyperparameter tuning was deemed unnecessary.

5.	Prediction
In the predictive phase of our project, the Gradient Boosting Classifier (GBC) was employed to forecast the nutritional category of items in the actual test dataset. The GBC demonstrated exceptional performance, achieving a remarkable 100% accuracy along with a perfect F1 score, precision, and recall across all classes. This level of accuracy is indicative of the model’s ability to generalize well from the training data to the test data, ensuring that each food item was categorized with utmost precision.

When deployed on unseen data, the model continued to showcase its robust predictive capabilities, yielding an impressive F1 score of 99.88%. These further cements the model's reliability and its suitability for the task at hand. The Gradient Boosting Classifier's consistent performance on both the test and unseen data sets a benchmark in predictive modeling for nutritional categorization.

Given these outcomes, Gradient Boosting Classifier stands as a highly competent model for categorizing nutritional content in line with the objectives of this project. The results affirm the model's capability to provide reliable and precise predictions, making it the best tool for dietary assessment and recommendations on the recipe website.

6.	Conclusion

Summary:
The core objective of this project was to construct a predictive model capable of classifying food items into nutritional categories - 'Nourishing', 'Indulgent', and 'Balanced'. The methodology embraced a structured data science pipeline: preprocessing the data, performing exploratory data analysis, preprocessing the data through cleaning and standardization, selecting pertinent features, and then training various classification models. Amongst these, the Gradient Boosting Classifier was ultimately chosen for its superior performance.

Findings:
The Gradient Boosting Classifier emerged as the standout model, demonstrating flawless accuracy and F1 scores in classifying the test data. This precision carried over to unseen data, with the model maintaining a near-perfect F1 score of 99.88%. These findings underscore the model's robustness and its ability to generalize beyond the training dataset.

Recommendations:
While the current model performs admirably, we recommend the following to ensure its ongoing relevancy and accuracy:

	Continuous Model Training: As new nutritional data becomes available, continuously train the model to adapt to new trends and information.
	Feature Exploration: Investigate additional features that could improve the model's predictive power, such as micronutrient content or glycemic indices.
	Model Interpretability: Implement techniques to enhance the interpretability of the model, aiding stakeholders in understanding the factors influencing predictions.
	Robust Validation: Utilize advanced cross-validation techniques to ensure the model's robustness.

7.	References
	Scikit-learn documentation.
	Gradient Boosting.
