import os
import shutil
from xml.sax.handler import all_features

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, accuracy_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import graphviz
from sklearn.tree import export_graphviz, export_text


# define
# =========================
ed_state = 0
num_experiments = 5
# model train param
test_size = 0.6
random_seed = 42
# =========================

qa_1_path = "../out/visualizations"


if ed_state == 0:
    import achieve_data
    if os.path.exists("../out"):
        shutil.rmtree("../out")
    os.makedirs("../out", exist_ok=True)
    os.makedirs(qa_1_path, exist_ok=True)
    achieve_data.mkdir(qa_1_path)


def data_cleaning(df):
    # 创建字典，将字符映射为数字
    sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    # 使用map函数将sex列中的字符替换为数字
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df

def analyze_and_visualize_abalone_data(data, output_dir):

    # Define age classes based on the rings
    age_bins = [0, 7, 10, 15, float('inf')]
    age_labels = ['Class 1: 0-7 years', 'Class 2: 8-10 years', 'Class 3: 11-15 years', 'Class 4: >15 years']
    data['Age Class'] = pd.cut(data['Rings'], bins=age_bins, labels=age_labels, right=True)

    # Update feature column names to match dataset
    feature_columns = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']

    # Define the visualization functions
    def visualize_age_class_distribution(data):
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Age Class', data=data, hue='Age Class', palette='viridis', dodge=False)
        plt.title('Distribution of Age Classes')
        plt.xlabel('Age Class')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/age_class_distribution.png")
        plt.close()

    def visualize_feature_distribution(data, feature_columns):
        plt.figure(figsize=(15, 10))
        data[feature_columns].hist(bins=20, figsize=(15, 10), layout=(3, 3), color='skyblue', edgecolor='black')
        plt.suptitle('Distribution of Features')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/feature_distribution.png")
        plt.close()

    def visualize_pairplot(data, feature_columns):
        pairplot_fig = sns.pairplot(data[feature_columns + ['Age Class']], hue='Age Class', palette='husl', plot_kws={'alpha': 0.5})
        pairplot_fig.fig.tight_layout()
        pairplot_fig.savefig(f"{output_dir}/pairplot.png")
        plt.close()

    def visualize_correlation_heatmap(data, feature_columns):
        plt.figure(figsize=(10, 8))
        sns.heatmap(data[feature_columns].corr(), annot=True, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Features')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/correlation_heatmap.png")
        plt.close()

    # Call the visualization functions
    visualize_age_class_distribution(data)
    visualize_feature_distribution(data, feature_columns)
    visualize_pairplot(data, feature_columns)
    visualize_correlation_heatmap(data, feature_columns)

    # Summary statistics for each feature
    summary_stats = data[feature_columns].describe()
    print("Summary Statistics for Each Feature:\n", summary_stats)

    # Save the summary statistics as a CSV file
    summary_stats.to_csv(os.path.join(output_dir, 'summary_statistics.csv'), index=True)


# Function to categorize Rings into Age Classes
def categorize_rings(df):
    # Define age classes based on the rings
    age_bins = [0, 7, 10, 15, float('inf')]
    age_labels = ['Class_1_0_7_years', 'Class_2_8_10_years', 'Class_3_11_15_years', 'Class_4_greater_15_years']
    df['Age Class'] = pd.cut(df['Rings'], bins=age_bins, labels=age_labels, right=True)
    return df

# Function to train and evaluate Decision Tree models with multiple experimental runs using different hyperparameters

def train_and_evaluate_unpruned_decision_tree(data, num_experiments, test_size):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']

    best_accuracy = 0
    best_model = None
    best_params = {}

    for i in range(num_experiments):  # Perform num_experiments experimental runs
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)

        # Randomly select hyperparameters for each experimental run
        max_depth = np.random.choice([None, 5, 10, 15, 20])
        min_samples_split = np.random.choice([2, 5, 10])
        min_samples_leaf = np.random.choice([1, 2, 4])

        # Initialize and train the Decision Tree classifier with selected hyperparameters
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_seed + i
        )
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # Evaluate performance
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        print(f"Run {i + 1}: Max Depth={max_depth}, Min Samples Split={min_samples_split}, Min Samples Leaf={min_samples_leaf}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}\n")

        # Keep track of the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = clf
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

    # Report the best model
    print("Best Model Testing Accuracy:")
    print(f"Best Testing Accuracy: {best_accuracy:.4f}")
    print(f"Best Hyperparameters: Max Depth={best_params['max_depth']}, Min Samples Split={best_params['min_samples_split']}, Min Samples Leaf={best_params['min_samples_leaf']}\n")

    # Visualize the best Decision Tree using Graphviz
    class_names = [str(label).replace(" ", "_").replace("<", "less_than_").replace(">", "greater_than_") for label in best_model.classes_]
    dot_data = export_graphviz(
        best_model,
        out_file=None,
        feature_names=feature_columns,
        class_names=class_names,
        filled=True,
        rounded=True,
        special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"{qa_1_path}/best_decision_tree", format='png', cleanup=True)

    # Translate selected nodes and leaves into IF and THEN rules
    tree_rules = export_text(best_model, feature_names=feature_columns)
    print("Rules from the best decision tree:\n")
    print(tree_rules)

    return best_model, X_train, X_test, y_train, y_test

# Function to train and evaluate Decision Tree models with pruning
def train_and_evaluate_pruned_decision_tree(data, num_experiments, test_size):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']

    best_accuracy = 0
    best_model = None
    best_params = {}

    for i in range(num_experiments):  # Perform num_experiments experimental runs
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)

        # Define different hyperparameters for each run
        max_depth = np.random.randint(3, 10)
        min_samples_split = np.random.randint(2, 10)
        min_samples_leaf = np.random.randint(1, 5)

        # Initialize and train the Decision Tree classifier with pruning
        clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, random_state=random_seed)
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)

        # Evaluate performance
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)

        print(f"Run {i + 1}: Max Depth={max_depth}, Min Samples Split={min_samples_split}, Min Samples Leaf={min_samples_leaf}")
        print(f"Training Accuracy: {train_accuracy:.4f}")
        print(f"Testing Accuracy: {test_accuracy:.4f}\n")

        # Keep track of the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_model = clf
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

    # Report the best model
    print("Best Pruned Model Parameters:")
    print(best_params)
    print(f"Best Testing Accuracy: {best_accuracy:.4f}\n")

    # Visualize the best pruned Decision Tree using Graphviz
    sanitized_class_names = [str(label).replace(" ", "_").replace(":", "").replace(">", "greater") for label in
                             best_model.classes_]
    dot_data = export_graphviz(best_model, out_file=None, feature_names=feature_columns,
                               class_names=sanitized_class_names, filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"{qa_1_path}/best_pruned_decision_tree", format='png', cleanup=True)

    return best_model, X_train, X_test, y_train, y_test

def apply_random_forests(data, max_num_trees, test_size, random_seed, output_dir):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    accuracies = []

    for n_estimators in range(1, max_num_trees + 1):
        # Initialize and train Random Forest model
        rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed, n_jobs=-1)
        rf_model.fit(X_train, y_train)

        # Predict on the test data
        y_pred = rf_model.predict(X_test)

        # Calculate accuracy
        accuracy = accuracy_score(y_test, y_pred)
        accuracies.append(accuracy)

        print(f"Number of Trees: {n_estimators}, Accuracy: {accuracy:.4f}")

    # Plotting accuracy vs. number of trees
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_num_trees + 1), accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Number of Trees in Ensemble')
    plt.ylabel('Accuracy Score')
    plt.title('Accuracy vs. Number of Trees in Random Forest')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/random_forest_accuracy.png")
    plt.close()

def train_and_evaluate_xgboost(data, test_size, random_seed, output_dir):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']

    # Convert y to numerical labels
    y = y.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Initialize and train the XGBoost model
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=random_seed)
    xgb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = xgb_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost Model Accuracy: {accuracy:.4f}\n")

    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_columns, xgb_model.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.title('XGBoost Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/xgboost_feature_importance.png")
    plt.close()

    return xgb_model

def train_and_evaluate_gradient_boosting(data, test_size, random_seed, output_dir):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']

    # Convert y to numerical labels
    y = y.cat.codes

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    # Initialize and train the Gradient Boosting model
    gb_model = GradientBoostingClassifier(random_state=random_seed)
    gb_model.fit(X_train, y_train)

    # Make predictions
    y_pred = gb_model.predict(X_test)

    # Evaluate performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Gradient Boosting Model Accuracy: {accuracy:.4f}\n")

    # Plotting feature importance
    plt.figure(figsize=(10, 6))
    plt.barh(feature_columns, gb_model.feature_importances_)
    plt.xlabel('Feature Importance')
    plt.title('Gradient Boosting Feature Importance')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/gradient_boosting_feature_importance.png")
    plt.close()

    return gb_model




def main():
    # 执行加载或更新data
    if ed_state == 0:
        achieve_data.achieve_data_main()

    # 从csv读取数据
    csv_file_path = "../data/abalone_data.csv"
    abalone = pd.read_csv(csv_file_path)

    # sex数据清洗
    abalone = data_cleaning(abalone)

    # Call the function with appropriate paths
    analyze_and_visualize_abalone_data(abalone, qa_1_path)

    print("\nTraining and Evaluating Unpruned Decision Tree:\n")
    best_unpruned_model, X_train, X_test, y_train, y_test = train_and_evaluate_unpruned_decision_tree(abalone, num_experiments, test_size)

    # Train and evaluate Decision Tree models with pruning
    print("\nTraining and Evaluating Pruned Decision Tree:\n")
    best_pruned_model, _, _, _, _ = train_and_evaluate_pruned_decision_tree(abalone, num_experiments, test_size)

    # Apply Random Forest classifier and evaluate performance
    print("\nApplying Random Forests and Evaluating Performance:\n")
    apply_random_forests(abalone, max_num_trees=50, test_size=test_size, random_seed=random_seed, output_dir=qa_1_path)

    # Train and evaluate XGBoost model
    print("\nTraining and Evaluating XGBoost Model:\n")
    xgb_model = train_and_evaluate_xgboost(abalone, test_size, random_seed, qa_1_path)

    # Train and evaluate Gradient Boosting model
    print("\nTraining and Evaluating Gradient Boosting Model:\n")
    gb_model = train_and_evaluate_gradient_boosting(abalone, test_size, random_seed, qa_1_path)



if __name__ == "__main__":
    main()
