# Required python packages:
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import model_selection
from xgboost import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

# Importing dataset:
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Percent of passengers that survived in our training data set vs percent of
# total survivors
y = train_data["Survived"].copy()
print('Train sample:    ', (sum(y)/len(y))*100, '%')
print('Total passanger: ', ((2224-1502)/2224)*100, '%')

# Prevewing the values in each feature
print(train_data.apply(lambda col: col.unique()))

# Remove useless features
train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)

#  Selecting features with missing values
NA = [(c, train_data[c].isna().mean()*100) for c in train_data]
NA = pd.DataFrame(NA, columns=["column_name", "percentage"])

# Display the percentage of missing values in each feature
NA = NA[NA.percentage > 0]
print(NA.sort_values("percentage", ascending=False))

# Remove the "Cabin" feature from the data sets
train_data.drop(['Cabin'], axis=1, inplace=True)
test_data.drop(['Cabin'], axis=1, inplace=True)

# Select the remaining features to replace with the most frequent value
columns_low_NA = ['Age', 'Embarked']

# Fill missing values for each feature with it's mean value
train_data[columns_low_NA] = train_data[columns_low_NA].fillna(train_data.mode(
).iloc[0])
test_data[columns_low_NA] = test_data[columns_low_NA].fillna(test_data.mode(
).iloc[0])

# Survival chart for comparison of each sex
sns.countplot(x="Sex", hue="Survived", data=train_data)
# plt.show()

# Survival chart for comparison for the "Parch" feature
sns.countplot(x="Parch", hue="Survived", data=train_data)
# plt.show()

# Survival chart for comparison for the "SibSp" feature
sns.countplot(x="SibSp", hue="Survived", data=train_data)
# plt.show()

# Percent of survival for each variable in "Embarked"
Embarked = train_data[['Embarked', 'Survived']].groupby([
    'Embarked'], as_index=False).mean().sort_values(
        by='Survived', ascending=False)
print(Embarked)

# Percent of survival for each variable in "Pclass"
Pclass = train_data[['Pclass', 'Survived']].groupby([
    'Pclass'], as_index=False).mean().sort_values(
        by='Survived', ascending=False)
print(Pclass)

# Survival chart for comparison of the "Age" feature
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Age', bins=30)
# plt.show()

# Survival chart for comparison of the "Fare" feature
g = sns.FacetGrid(train_data, col='Survived')
g.map(plt.hist, 'Fare', bins=25)
# plt.show()

# Feature mapping for the feature "Pclass"
Pclass = {1: 'PclassA', 2: 'PclassB', 3: 'PclassC'}
train_data['Pclass'] = train_data['Pclass'].map(Pclass)
test_data['Pclass'] = test_data['Pclass'].map(Pclass)

# Changing feature "sex" from categorical to numerical
sex_bin = {"female": 0,   # Zero is female
           "male": 1}     # One is for male
train_data['Sex'] = train_data['Sex'].map(sex_bin)
test_data['Sex'] = test_data['Sex'].map(sex_bin)

# One-hot encoding for the features "Embarked" and "Pclass"
train_encoded = pd.get_dummies(train_data, columns=['Embarked', 'Pclass'])
test_encoded = pd.get_dummies(test_data, columns=['Embarked', 'Pclass'])

# Drop the target variable for testing
X = train_encoded.drop(['Survived'], axis=1).copy()

# Split data into training and validation data, for both features and target
X_train, X_test, y_train, y_test = train_test_split(
                                            X, y, random_state=0, stratify=y)
X, y = make_classification(random_state=0)

# Set up model with XGboost
model = xgb.XGBClassifier(
    seed=0, max_depth=6, subsample=1, n_estimators=100, learning_rate=0.3,
    min_child_weight=1, random_state=5, reg_alpha=0, reg_lambda=1,
    use_label_encoder=False)
# Fit the model
model.fit(X_train, y_train)
y_predict = model.predict(X_test)
y_train_predict = model.predict(X_train)
# Print test and training accuracy scores
print('Train accurcy', (accuracy_score(y_train, y_train_predict))*100, '%')
print('Test accurcy ', (accuracy_score(y_test, y_predict))*100, '%')

# Confusion Matrix Diagram
plot_confusion_matrix(model, X_test, y_test, display_labels=[
    "Did not Survive", "Survived"])
plt.title("confusion Matrix Diagram")


# Plot of visual importance of each feature
feature_imp = pd.Series(
    model.feature_importances_, index=X_train.columns).sort_values(
                                                            ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=feature_imp, y=feature_imp.index)
# Add labels to graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features")
plt.tight_layout()


# Plot of single tree
plot_tree(model, num_trees=0)
plt.gcf().set_size_inches(30, 30)
plt.title("Plot of Single Tree in our Model")
plt.show()
