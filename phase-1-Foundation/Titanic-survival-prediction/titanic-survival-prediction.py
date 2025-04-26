import pandas as pd                      # For data manipulation
import seaborn as sns                    # For data visualization
import matplotlib.pyplot as plt          # For plotting
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.linear_model import LogisticRegression    # The model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # For evaluation

# Load the dataset
df = pd.read_csv("https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv")

# View first few rows
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Drop columns we won't use (PassengerId, Name, Ticket, Cabin)
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Fill missing Age values with median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill missing Embarked values with mode
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Convert categorical columns to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True)

# View the cleaned dataset
print(df.head())

# Define features (X) and target (y)
X = df.drop('Survived', axis=1)   # Features: all except 'Survived'
y = df['Survived']                # Target: whether survived (0 or 1)

# Split into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)  # increase max_iter for convergence

# Train the model
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Optional: visualize the confusion matrix using heatmap
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Reds')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
