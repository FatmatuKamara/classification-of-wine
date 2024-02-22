import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
#load the wine dataset
wine_data = load_wine()
#print the description of the dataset
print("wine dataset description:")
print(wine_data.DESCR)
#Extract features
features = wine_data.feature_names
#print the features of the wine dataset
print("\nFeatures of the wine Dataset:")
for i, features in enumerate(features):
    print(f"Feature {i+1}: {features}")

    # Explore dataset structure
    print("Dataset keys:", wine_data.keys())
    print("Feature names:", wine_data['feature_names'])
    print("Target names:", wine_data['target_names'])
    print("Number of samples:", len(wine_data['data']))
    print("Number of features:", len(wine_data['feature_names']))

    # Task 3: Data Preprocessing
    import pandas as pd

    # Create DataFrame from features and target
    wine_df = pd.DataFrame(wine_data['data'], columns=wine_data['feature_names'])
    wine_df['target'] = wine_data['target']

    # Check for missing values
    print("Missing values:\n", wine_df.isnull().sum())

    # Split features and target labels
    X = wine_df.drop('target', axis=1)
    y = wine_df['target']
    # Task 4: Train-Test Split
    from sklearn.model_selection import train_test_split

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Task 5: Model 1 - Logistic Regression
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score

    # Instantiate logistic regression model
    logistic_model = LogisticRegression(max_iter=10000)

    # Train the model
    logistic_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_logistic = logistic_model.predict(X_test)

    # Calculate accuracy
    accuracy_logistic = accuracy_score(y_test, y_pred_logistic)
    print("Logistic Regression Accuracy:", accuracy_logistic)

    # Task 6: Model 2 - Decision Trees
    from sklearn.tree import DecisionTreeClassifier

    # Instantiate decision tree model
    tree_model = DecisionTreeClassifier()

    # Train the model
    tree_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_tree = tree_model.predict(X_test)

    # Calculate accuracy
    accuracy_tree = accuracy_score(y_test, y_pred_tree)
    print("Decision Tree Accuracy:", accuracy_tree)

    # Task 7: Model 3 - Support Vector Machines (SVM)
    from sklearn.svm import SVC

    # Instantiate SVM model
    svm_model = SVC()

    # Train the model
    svm_model.fit(X_train, y_train)

    # Predict on the test set
    y_pred_svm = svm_model.predict(X_test)

    # Calculate accuracy
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    print("SVM Accuracy:", accuracy_svm)

    # Task 8: Model Evaluation
    from sklearn.metrics import confusion_matrix, classification_report

    # Evaluate Logistic Regression
    print("\nLogistic Regression:")
    print(confusion_matrix(y_test, y_pred_logistic))
    print(classification_report(y_test, y_pred_logistic))

    # Evaluate Decision Trees
    print("\nDecision Tree:")
    print(confusion_matrix(y_test, y_pred_tree))
    print(classification_report(y_test, y_pred_tree))

    # Evaluate SVM
    print("\nSVM:")
    print(confusion_matrix(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm))

from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'penalty': ['l1', 'l2'],  # Regularization type
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Inverse of regularization strength
    'solver': ['liblinear', 'saga']  # Algorithm to use in optimization problem
}

# Instantiate logistic regression model
logistic_model = LogisticRegression(max_iter=10000)

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=logistic_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit grid search to the data
grid_search.fit(X_train, y_train)

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use best model for prediction
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy:", accuracy_best)
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'criterion': ['gini', 'entropy'],  # Split criterion
    'max_depth': [None, 5, 10, 15, 20],  # Maximum depth of the tree
    'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
    'min_samples_leaf': [1, 2, 4]  # Minimum number of samples required to be at a leaf node
}

# Instantiate decision tree model
tree_model = DecisionTreeClassifier()

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=tree_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit grid search to the data
grid_search.fit(X_train, y_train)

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use best model for prediction
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy:", accuracy_best)
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

# Define hyperparameters to tune
param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization parameter
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],  # Kernel type
    'gamma': ['scale', 'auto']  # Kernel coefficient
}

# Instantiate SVM model
svm_model = SVC()

# Grid search with cross-validation
grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit grid search to the data
grid_search.fit(X_train, y_train)

# Get best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

# Use best model for prediction
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)

# Evaluate best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print("Best Model Accuracy:", accuracy_best)

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate ROC curve and ROC area for each class
for i in range(len(wine_data.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_logistic, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(wine_data.target_names)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Logistic Regression Model')
plt.legend(loc="lower right")
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate ROC curve and ROC area for each class
for i in range(len(wine_data.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_pred_tree, pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(wine_data.target_names)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - Decision Tree Model')
plt.legend(loc="lower right")
plt.show()
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()

# Calculate decision function scores
y_score = best_model.decision_function(X_test)

# Calculate ROC curve and ROC area for each class
for i in range(len(wine_data.target_names)):
    fpr[i], tpr[i], _ = roc_curve(y_test, y_score[:, i], pos_label=i)
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot ROC curves
plt.figure(figsize=(8, 6))
for i in range(len(wine_data.target_names)):
    plt.plot(fpr[i], tpr[i], label=f'ROC curve (class {i}) (area = {roc_auc[i]:0.2f})')

plt.plot([0, 1], [0, 1], 'k--')  # Plot diagonal line
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve - SVM Model')
plt.legend(loc="lower right")
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred_logistic)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
            xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Logistic Regression Model')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
            xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Decision Tree Model')
plt.show()

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g',
            xticklabels=wine_data.target_names, yticklabels=wine_data.target_names)
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Support Vector Machine Model')
plt.show()

