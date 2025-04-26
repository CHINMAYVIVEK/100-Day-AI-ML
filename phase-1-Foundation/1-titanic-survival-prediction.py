import pandas as pd                      # For data manipulation
import seaborn as sns                    # For data visualization
import matplotlib.pyplot as plt           # For plotting graphs
from sklearn.model_selection import train_test_split  # To split the data into training and test sets
from sklearn.linear_model import LogisticRegression    # For Logistic Regression model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For model evaluation

# Step 1: Load the Titanic dataset from a URL
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# Step 2: Inspect the first few rows of the dataset to understand its structure
print("Dataset Preview:")
print(df.head())

# Step 3: Check for missing values in each column
# Understanding missing data is crucial as we need to decide how to handle it
print("\nMissing Values per Column:")
print(df.isnull().sum())

# Step 4: Data Cleaning - Dropping unnecessary columns
# The columns like 'PassengerId', 'Name', 'Ticket', and 'Cabin' are not useful for the prediction.
# We'll drop them to simplify the model.
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Step 5: Handle missing values
# - We will fill missing values in 'Age' with the median since it's a numerical column.
# - We will fill missing values in 'Embarked' with the mode (most frequent value) because it's categorical.
df['Age'].fillna(df['Age'].median(), inplace=True)  # Median is robust to outliers
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)  # Mode (most frequent) for categorical data

# Step 6: Convert categorical variables into numerical format using one-hot encoding
# 'Sex' and 'Embarked' are categorical variables. We'll use pandas get_dummies to create binary columns for each category.
# We drop the first category in each column to avoid multicollinearity (dummy variable trap).
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# Step 7: Inspect the cleaned dataset
print("\nCleaned Dataset Preview:")
print(df.head())

# Step 8: Define features (X) and target (y)
# 'Survived' is the target variable we are trying to predict.
# All other columns will be our features (independent variables).
X = df.drop('Survived', axis=1)   # Features: all columns except 'Survived'
y = df['Survived']                # Target: whether the passenger survived (1) or not (0)

# Step 9: Split the data into training and testing sets
# We'll use 80% of the data for training and 20% for testing. This allows us to evaluate the model's performance.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Initialize the Logistic Regression model
# Logistic Regression is a linear model used for binary classification tasks.
# We increase the max_iter parameter to ensure the model converges during training.
model = LogisticRegression(max_iter=1000)

# Step 11: Train the Logistic Regression model on the training data
# Fitting the model to the training data so it can learn the relationship between the features and the target.
model.fit(X_train, y_train)

# Step 12: Make predictions on the test data
# After the model is trained, we can predict whether passengers in the test set survived or not.
y_pred = model.predict(X_test)

# Step 13: Evaluate the model
# - Accuracy: The percentage of correctly predicted instances out of all instances.
# - Confusion Matrix: A table to visualize the performance of the model, showing true positives, false positives, etc.
# - Classification Report: Includes precision, recall, f1-score, and support for each class.
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Step 14: Visualizing the confusion matrix using a heatmap for better interpretation
# A confusion matrix helps us see how well the model is performing, showing the true positives, true negatives, 
# false positives, and false negatives.
plt.figure(figsize=(7, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds', cbar=False)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix of Titanic Survival Prediction')
plt.show()

# Optional: Visualizing the distribution of key features
# Let's visualize the distribution of the features like 'Age', 'Fare', and the 'Survived' variable.
plt.figure(figsize=(10, 6))

# Plot the distribution of Age for survivors and non-survivors
plt.subplot(1, 2, 1)
sns.histplot(df[df['Survived'] == 1]['Age'], kde=True, color='green', label='Survived')
sns.histplot(df[df['Survived'] == 0]['Age'], kde=True, color='red', label='Did not survive')
plt.title('Age Distribution by Survival')
plt.legend()

# Plot the distribution of Fare for survivors and non-survivors
plt.subplot(1, 2, 2)
sns.histplot(df[df['Survived'] == 1]['Fare'], kde=True, color='green', label='Survived')
sns.histplot(df[df['Survived'] == 0]['Fare'], kde=True, color='red', label='Did not survive')
plt.title('Fare Distribution by Survival')
plt.legend()

plt.tight_layout()
plt.show()