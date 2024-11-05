import os
import shutil

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, roc_auc_score, roc_curve, accuracy_score, f1_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import random

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
import graphviz
from sklearn.tree import export_graphviz, export_text
from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import layers, models



# define
# =========================
ed_state = 0
num_experiments = 5
# model train param
test_size = 0.6
random_seed = 42
# =========================

qa_1_path = "../out/q_a/visualizations"
qa_2_path = "../results/q_a"
results_file_path = f"{qa_2_path}/model_results.csv"
best_results_file_path = f"{qa_2_path}/best_model_results.csv"

if ed_state == 0:
    import achieve_data

def create_dir():


    if os.path.exists("../out"):
        shutil.rmtree("../out")
    if os.path.exists("../results"):
        shutil.rmtree("../results")
    os.makedirs("../out", exist_ok=True)
    os.makedirs("../out/q_a/model_results", exist_ok=True)
    os.makedirs(qa_1_path, exist_ok=True)
    os.makedirs(qa_2_path, exist_ok=True)
    # achieve_data.mkdir(qa_1_path)

def data_cleaning(df):
    # 创建字典，将字符映射为数字
    sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    # 使用map函数将sex列中的字符替换为数字
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df

def encode_target(y):
    target_mapping = {
        'Class_1_0_7_years': 0,
        'Class_2_8_10_years': 1,
        'Class_3_11_15_years': 2,
        'Class_4_greater_15_years': 3
    }
    return y.map(target_mapping)

def analyze_and_visualize_abalone_data(data, output_dir):
    # Define age classes based on the rings
    age_bins = [-1, 7, 10, 15, float('inf')]
    age_labels = ['Class_1_0_7_years', 'Class_2_8_10_years', 'Class_3_11_15_years', 'Class_4_greater_15_years']
    data['Age Class'] = pd.cut(data['Rings'], bins=age_bins, labels=age_labels, right=True)

    # Update feature column names to match dataset
    feature_columns = ['Sex','Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']

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

def select_features_and_target(data):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']

    # X = data.iloc[:, :-1]
    # y = data.iloc[:, -1]
    return X, y

def save_results_to_csv(results, output_file):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)

def train_and_evaluate_basic_decision_tree(X, y, num_experiments, test_size, random_seed, output_dir):
    # Store performance metrics for each experiment
    experiment_metrics = []

    best_accuracy = 0
    best_auc = 0
    best_f1 = 0
    best_model = None
    best_params = {}

    for i in range(num_experiments):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)

        # Define hyperparameters with pre-pruning
        max_depth = np.random.randint(3, 10)
        min_samples_split = np.random.randint(2, 10)
        min_samples_leaf = np.random.randint(1, 5)

        # Initialize and train the Decision Tree classifier
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_seed + i
        )
        clf.fit(X_train, y_train)

        # Make predictions
        y_pred_test = clf.predict(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        auc = roc_auc_score(y_test, clf.predict_proba(X_test), multi_class='ovr')

        # Store experiment metrics
        experiment_metrics.append({
            'Experiment': i + 1,
            'Max Depth': max_depth,
            'Min Samples Split': min_samples_split,
            'Min Samples Leaf': min_samples_leaf,
            'Test Accuracy': test_accuracy,
            'F1 Score': f1,
            'AUC': auc
        })

        # Track the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_auc = auc
            best_f1 = f1
            best_model = clf
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

    # Visualize the best Decision Tree and print IF-THEN rules
    dot_data = export_graphviz(
        best_model, feature_names=X.columns, class_names=best_model.classes_.astype(str),
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"{output_dir}/best_basic_tree", format='png', cleanup=True)

    print(f"Best Model Parameters (Basic with Pre-Pruning): {best_params}")
    print(f"Best Model Test Accuracy (Basic): {best_accuracy:.4f}")
    print("IF-THEN rules for the best basic decision tree:\n")
    print(export_text(best_model, feature_names=X.columns))

    # Compute summary statistics
    metrics_df = pd.DataFrame(experiment_metrics)
    summary_statistics = metrics_df[['Test Accuracy', 'F1 Score', 'AUC']].agg(['mean', 'var', 'std']).T

    # Return metrics, summary statistics, best model, best parameters, and best scores
    return metrics_df, summary_statistics, best_model, best_params, best_accuracy, best_auc, best_f1

def train_and_evaluate_pruned_postprocessed_decision_tree(X, y, num_experiments, test_size, random_seed, output_dir):
    # Store performance metrics for each experiment
    experiment_metrics = []

    best_accuracy = 0
    best_auc = 0
    best_f1 = 0
    best_model = None
    best_params = {}

    for i in range(num_experiments):
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)

        # Define hyperparameters with pre-pruning
        max_depth = np.random.randint(3, 10)
        min_samples_split = np.random.randint(2, 10)
        min_samples_leaf = np.random.randint(1, 5)

        # Initialize and train the Decision Tree classifier
        clf = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_seed + i
        )
        clf.fit(X_train, y_train)

        # Post-process pruning using cost complexity pruning
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas
        pruned_clf = None
        best_pruned_score = 0

        for ccp_alpha in ccp_alphas:
            temp_clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                random_state=random_seed + i
            )
            temp_clf.fit(X_train, y_train)
            temp_score = temp_clf.score(X_test, y_test)
            if temp_score > best_pruned_score:
                best_pruned_score = temp_score
                pruned_clf = temp_clf

        # Make predictions
        y_pred_test = pruned_clf.predict(X_test)

        # Calculate metrics
        test_accuracy = accuracy_score(y_test, y_pred_test)
        f1 = f1_score(y_test, y_pred_test, average='weighted')
        auc = roc_auc_score(y_test, pruned_clf.predict_proba(X_test), multi_class='ovr')

        # Store experiment metrics
        experiment_metrics.append({
            'Experiment': i + 1,
            'Max Depth': max_depth,
            'Min Samples Split': min_samples_split,
            'Min Samples Leaf': min_samples_leaf,
            'Test Accuracy': test_accuracy,
            'F1 Score': f1,
            'AUC': auc
        })

        # Track the best model
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            best_auc = auc
            best_f1 = f1
            best_model = pruned_clf
            best_params = {
                'max_depth': max_depth,
                'min_samples_split': min_samples_split,
                'min_samples_leaf': min_samples_leaf
            }

    # Visualize the best Pruned Decision Tree and print IF-THEN rules
    dot_data = export_graphviz(
        best_model, feature_names=X.columns, class_names=best_model.classes_.astype(str),
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"{output_dir}/best_pruned_postprocessed_tree", format='png', cleanup=True)

    print(f"Best Model Parameters (Pruned and Post-processed): {best_params}")
    print(f"Best Model Test Accuracy (Pruned and Post-processed): {best_accuracy:.4f}")
    print("IF-THEN rules for the best pruned and post-processed decision tree:\n")
    print(export_text(best_model, feature_names=X.columns))

    # Compute summary statistics
    metrics_df = pd.DataFrame(experiment_metrics)
    summary_statistics = metrics_df[['Test Accuracy', 'F1 Score', 'AUC']].agg(['mean', 'var', 'std']).T

    # Return metrics, summary statistics, best model, best parameters, and best scores
    return metrics_df, summary_statistics, best_model, best_params, best_accuracy, best_auc, best_f1

def train_and_evaluate_model(model, X, y, num_experiments, random_seed):

    accuracies, aucs, f1_scores = [], [], []
    best_accuracy, best_auc, best_f1 = 0, 0, 0


    for i in range(num_experiments):

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=i + random_seed)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred), multi_class='ovr')
        f1 = f1_score(y_test, y_pred, average='weighted')

        accuracies.append(accuracy)
        aucs.append(auc)
        f1_scores.append(f1)

        # Update best values if current experiment scores are higher
        if accuracy > best_accuracy:
            best_accuracy = accuracy
        if auc > best_auc:
            best_auc = auc
        if f1 > best_f1:
            best_f1 = f1

    # Calculate means, standard deviations, and variances
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    var_accuracy = np.var(accuracies)

    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)
    var_auc = np.var(aucs)

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    var_f1 = np.var(f1_scores)

    return (mean_accuracy, std_accuracy, var_accuracy, mean_auc, std_auc, var_auc,
            mean_f1, std_f1, var_f1, best_accuracy, best_auc, best_f1)

def compare_l2_and_dropout(X, y, num_experiments, test_size, random_seed, output_dir):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed)

    y_train = y_train.astype('int')
    y_test = y_test.astype('int')

    # Define hyperparameter combinations
    hyperparameter_combinations = [
        {"dropout_rate": 0.2, "weight_decay": 0.01},
        {"dropout_rate": 0.3, "weight_decay": 0.001},
        {"dropout_rate": 0.5, "weight_decay": 0.0001}
    ]

    results = []

    for i, params in enumerate(hyperparameter_combinations):
        dropout_rate = params["dropout_rate"]
        weight_decay = params["weight_decay"]

        # L2 Regularization Model
        l2_model = MLPClassifier(
            hidden_layer_sizes=(100,),
            solver='adam',
            alpha=weight_decay,  # L2 regularization strength
            max_iter=500,
            random_state=random_seed
        )
        l2_model.fit(X_train, y_train)
        y_pred_l2 = l2_model.predict(X_test)
        l2_accuracy = accuracy_score(y_test, y_pred_l2)
        l2_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_l2), multi_class='ovr')
        l2_f1 = f1_score(y_test, y_pred_l2, average='weighted')

        # Dropout Model using TensorFlow
        dropout_model = tf.keras.Sequential([
            layers.InputLayer(shape=(X_train.shape[1],)),
            layers.Dense(100, activation='relu'),
            layers.Dropout(dropout_rate),  # Dropout rate
            layers.Dense(4, activation='softmax')
        ])
        dropout_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        dropout_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        y_pred_dropout = np.argmax(dropout_model.predict(X_test), axis=1)
        dropout_accuracy = accuracy_score(y_test, y_pred_dropout)
        dropout_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_dropout), multi_class='ovr')
        dropout_f1 = f1_score(y_test, y_pred_dropout, average='weighted')

        # Append results for each combination
        results.append({
            "Combination": f"Dropout rate: {dropout_rate}, Weight decay (λ): {weight_decay}",
            "L2 Accuracy": l2_accuracy, "L2 AUC": l2_auc, "L2 F1": l2_f1,
            "Dropout Accuracy": dropout_accuracy, "Dropout AUC": dropout_auc, "Dropout F1": dropout_f1
        })

        print(f"Combination {i + 1}: Dropout rate {dropout_rate}, Weight decay {weight_decay}")
        print(f"L2 Model - Accuracy: {l2_accuracy:.4f}, AUC: {l2_auc:.4f}, F1 Score: {l2_f1:.4f}")
        print(f"Dropout Model - Accuracy: {dropout_accuracy:.4f}, AUC: {dropout_auc:.4f}, F1 Score: {dropout_f1:.4f}")

    # Convert to DataFrame for easier saving and display
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(qa_2_path, "l2_vs_dropout_results.csv"), index=False)
    print("L2 vs Dropout comparison results saved to l2_vs_dropout_results.csv")

    return results_df

def calculate_and_append_summary(summary_statistics, results, model_name):
    """
    Calculates summary statistics (mean, variance, std) for each metric in metrics_df
    and appends the results to the provided results list.

    Parameters:
    - metrics_df (pd.DataFrame): DataFrame containing experiment metrics with columns for each metric.
    - results (list): List to which the summary statistics will be appended.
    - model_name (str): Name of the model for labeling in the results.

    Returns:
    - None: The function modifies the results list in place.
    """
    # Calculate summary statistics (mean, variance, std) for selected metrics
    # summary_statistics = metrics_df[['Test Accuracy', 'F1 Score', 'AUC']].agg(['mean', 'var', 'std']).T

    # Extract mean, variance, and std for each metric
    mean_accuracy = summary_statistics.loc['Test Accuracy', 'mean']
    var_accuracy = summary_statistics.loc['Test Accuracy', 'var']
    std_accuracy = summary_statistics.loc['Test Accuracy', 'std']

    mean_auc = summary_statistics.loc['AUC', 'mean']
    var_auc = summary_statistics.loc['AUC', 'var']
    std_auc = summary_statistics.loc['AUC', 'std']

    mean_f1 = summary_statistics.loc['F1 Score', 'mean']
    var_f1 = summary_statistics.loc['F1 Score', 'var']
    std_f1 = summary_statistics.loc['F1 Score', 'std']

    # Append to results with extracted summary statistics
    results.append({
        "Model": model_name,
        "Mean Accuracy": mean_accuracy,
        "Variance Accuracy": var_accuracy,
        "Std Accuracy": std_accuracy,
        "Mean AUC": mean_auc,
        "Variance AUC": var_auc,
        "Std AUC": std_auc,
        "Mean F1 Score": mean_f1,
        "Variance F1 Score": var_f1,
        "Std F1 Score": std_f1
    })

def main():
    create_dir()
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

    results = []
    best_results = []

    X, y =  select_features_and_target(abalone)
    y = encode_target(y)  # Encode target labels


    # 运行无剪枝决策树并保存结果
    unpruned_metrics_df, unpruned_summary, unpruned_best_model, unpruned_best_params, unpruned_best_accuracy, unpruned_best_auc, unpruned_best_f1 = train_and_evaluate_basic_decision_tree(
        X, y, num_experiments, test_size, random_seed, qa_2_path
    )
    unpruned_metrics_df.to_csv(f"{qa_2_path}/unpruned_tree_experiment_metrics.csv", index=False)
    # unpruned_summary.to_csv(f"{qa_2_path}/unpruned_tree_summary_statistics.csv", index=True)

    # Assuming metrics_df is calculated for a specific model
    calculate_and_append_summary(unpruned_summary, results, "Unpruned Decision Tree")

    # 将无剪枝决策树的最佳结果添加到列表中
    best_results.append({
        "Model": "Unpruned Decision Tree",
        "Best Accuracy": unpruned_best_accuracy,
        "Best AUC": unpruned_best_auc,
        "Best F1 Score": unpruned_best_f1
    })

    # 运行剪枝决策树并保存结果
    pruned_metrics_df, pruned_summary, pruned_best_model, pruned_best_params, pruned_best_accuracy, pruned_best_auc, pruned_best_f1 = train_and_evaluate_pruned_postprocessed_decision_tree(
        X, y, num_experiments, test_size, random_seed, qa_2_path
    )
    pruned_metrics_df.to_csv(f"{qa_2_path}/pruned_tree_experiment_metrics.csv", index=False)
    # pruned_summary.to_csv(f"{qa_2_path}/pruned_tree_summary_statistics.csv", index=True)

    # Assuming metrics_df is calculated for a specific model
    calculate_and_append_summary(pruned_summary, results, "Pruned Decision Tree")

    # 将剪枝决策树的最佳结果添加到列表中
    best_results.append({
        "Model": "Pruned Decision Tree",
        "Best Accuracy": pruned_best_accuracy,
        "Best AUC": pruned_best_auc,
        "Best F1 Score": pruned_best_f1
    })


    # Define models and parameters
    models = [
        ("Random Forest", RandomForestClassifier(n_estimators=100, random_state=random_seed)),
        ("XGBoost", XGBClassifier(eval_metric='mlogloss', random_state=random_seed)),
        ("Gradient Boosting", GradientBoostingClassifier(random_state=random_seed)),
        ("Neural Network (Adam)", MLPClassifier(hidden_layer_sizes=(100,), solver='adam', max_iter=500, random_state=random_seed)),
        ("Neural Network (SGD)", MLPClassifier(hidden_layer_sizes=(100,), solver='sgd', max_iter=500, random_state=random_seed))
    ]

    for model_name, model in models:
        (mean_accuracy, std_accuracy, var_accuracy, mean_auc, std_auc, var_auc,
         mean_f1, std_f1, var_f1, best_accuracy, best_auc, best_f1) = train_and_evaluate_model(model, X, y, num_experiments, random_seed)

        results.append({
            "Model": model_name,
            "Mean Accuracy": mean_accuracy,
            "Variance Accuracy": var_accuracy,
            "Std Accuracy": std_accuracy,
            "Mean AUC": mean_auc,
            "Variance AUC": var_auc,
            "Std AUC": std_auc,
            "Mean F1 Score": mean_f1,
            "Variance F1 Score": var_f1,
            "Std F1 Score": std_f1
        })

        best_results.append({
            "Model": model_name,
            "Best Accuracy": best_accuracy,
            "Best AUC": best_auc,
            "Best F1 Score": best_f1
        })

        print(
            f"{model_name} - Mean Accuracy: {mean_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}, Mean AUC: {mean_auc:.4f}, Std AUC: {std_auc:.4f}, Mean F1 Score: {mean_f1:.4f}, Std F1 Score: {std_f1:.4f}")
        print(f"{model_name} - Best Accuracy: {best_accuracy:.4f}, Best AUC: {best_auc:.4f}, Best F1 Score: {best_f1:.4f}")

    # L2

    # L2 vs Dropout comparison with multiple hyperparameter combinations
    l2_dropout_results = compare_l2_and_dropout(X, y, num_experiments, test_size, random_seed, qa_2_path)

    # Append L2 vs Dropout best results for summary
    l2_best_accuracy = max(l2_dropout_results["L2 Accuracy"].max(), l2_dropout_results["Dropout Accuracy"].max())
    l2_best_auc = max(l2_dropout_results["L2 AUC"].max(), l2_dropout_results["Dropout AUC"].max())
    l2_best_f1 = max(l2_dropout_results["L2 F1"].max(), l2_dropout_results["Dropout F1"].max())

    best_results.append({
        "Model": "L2 vs Dropout",
        "Best Accuracy": l2_best_accuracy,
        "Best AUC": l2_best_auc,
        "Best F1 Score": l2_best_f1
    })


    # Save all results to CSV
    save_results_to_csv(results, results_file_path)
    save_results_to_csv(best_results, best_results_file_path)
    print(f"Mean results saved to {results_file_path}")
    print(f"Best results saved to {best_results_file_path}")

if __name__ == "__main__":
    main()