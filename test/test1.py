import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neural_network import MLPClassifier
import xgboost as xgb

# Load the dataset
data_url = "http://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
column_names = [
    'Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings'
]
df = pd.read_csv(data_url, names=column_names)


# Preprocess the dataset
def preprocess_data(df):
    # Convert 'Sex' column to numeric
    df['Sex'] = df['Sex'].map({'M': 0, 'F': 1, 'I': 2})

    # Create age classes
    df['AgeClass'] = pd.cut(df['Rings'], bins=[0, 7, 10, 15, np.inf], labels=[0, 1, 2, 3])
    df['AgeClass'] = df['AgeClass'].astype(int)

    # Drop the original 'Rings' column
    df = df.drop(columns=['Rings'])
    return df


df = preprocess_data(df)


# EDA
def perform_eda(df):
    sns.countplot(x='AgeClass', data=df)
    plt.title('Distribution of Age Classes')
    plt.savefig('distribution_age_classes.png')
    plt.clf()

    df.hist(figsize=(10, 10))
    plt.suptitle('Feature Distributions')
    plt.savefig('feature_distributions.png')
    plt.clf()


perform_eda(df)

# Split data into features and target
X = df.drop(columns=['AgeClass'])
y = df['AgeClass']

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Decision Tree Classifier
def train_decision_tree(X_train, X_test, y_train, y_test):
    depths = [3, 5, 7, 9, 11]
    best_tree = None
    best_score = 0

    for depth in depths:
        clf = DecisionTreeClassifier(max_depth=depth, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        score = accuracy_score(y_test, y_pred)

        if score > best_score:
            best_score = score
            best_tree = clf

    # Visualize the best tree
    plt.figure(figsize=(20, 10))
    plot_tree(best_tree, filled=True, feature_names=X.columns, class_names=['0', '1', '2', '3'])
    plt.savefig('best_decision_tree.png')
    plt.clf()

    return best_tree


best_tree = train_decision_tree(X_train, X_test, y_train, y_test)


# Random Forest Classifier
def train_random_forest(X_train, X_test, y_train, y_test):
    n_estimators = [10, 50, 100, 150]

    for n in n_estimators:
        rf = RandomForestClassifier(n_estimators=n, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        score = accuracy_score(y_test, y_pred)


train_random_forest(X_train, X_test, y_train, y_test)


# XGBoost and Gradient Boosting Classifier
def train_boosting_models(X_train, X_test, y_train, y_test):
    # XGBoost
    xgb_clf = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)

    # Gradient Boosting
    gb_clf = GradientBoostingClassifier(random_state=42)
    gb_clf.fit(X_train, y_train)
    y_pred_gb = gb_clf.predict(X_test)


train_boosting_models(X_train, X_test, y_train, y_test)


# Simple Neural Network Classifier (MLP)
def train_neural_network(X_train, X_test, y_train, y_test):
    mlp_clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, solver='adam', random_state=42)
    mlp_clf.fit(X_train, y_train)
    y_pred_mlp = mlp_clf.predict(X_test)


train_neural_network(X_train, X_test, y_train, y_test)


# Comparison of L2 Regularization and Dropout (MLP)
def train_nn_with_regularization(X_train, X_test, y_train, y_test):
    params = [
        {'alpha': 0.0001, 'dropout': 0.2},
        {'alpha': 0.001, 'dropout': 0.3},
        {'alpha': 0.01, 'dropout': 0.5},
    ]

    for param in params:
        mlp_clf = MLPClassifier(hidden_layer_sizes=(50,), max_iter=500, solver='adam', alpha=param['alpha'],
                                random_state=42)
        mlp_clf.fit(X_train, y_train)
        y_pred_mlp = mlp_clf.predict(X_test)


train_nn_with_regularization(X_train, X_test, y_train, y_test)
