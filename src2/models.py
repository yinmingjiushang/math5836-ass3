from sklearn.neural_network import MLPClassifier
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split, ParameterSampler
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.tree import export_graphviz
import graphviz
import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

from sklearn.utils.class_weight import compute_class_weight


import out_path


def train_and_evaluate_basic_decision_tree(X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None):

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
            random_state=random_seed + i,
            class_weight=class_weights  # Use the externally provided class weights
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

    # Visualize the best Decision Tree and output IF-THEN rules to a file
    dot_data = export_graphviz(
        best_model, feature_names=X.columns, class_names=best_model.classes_.astype(str),
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"{output_dir}/best_basic_tree", format='png', cleanup=True)

    print("================================================================================")
    print(f"Best Model Parameters (Basic with Pre-Pruning): {best_params}")
    print(f"Best Model Test Accuracy (Basic): {best_accuracy:.4f}")

    # Output IF-THEN rules to a file
    ifthen_rules = export_text(best_model, feature_names=list(X.columns))
    with open(f"{output_dir}/best_basic_tree_rules.txt", "w") as file:
        file.write("IF-THEN rules for the best basic decision tree:\n")
        file.write(ifthen_rules)
    print("IF-THEN rules for the best basic decision tree have been saved to file.")

    # Feature Importance
    feature_importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv(f"{output_dir}/best_basic_tree_feature_importance.csv", index=False)
    print("Feature importance for the best basic decision tree has been saved to CSV.")

    # Compute summary statistics
    metrics_df = pd.DataFrame(experiment_metrics)
    summary_statistics = metrics_df[['Test Accuracy', 'F1 Score', 'AUC']].agg(['mean', 'var', 'std']).T

    # Return metrics, summary statistics, best model, best parameters, best scores, and feature importances
    return metrics_df, summary_statistics, best_model, best_params, best_accuracy, best_auc, best_f1

def train_and_evaluate_pruned_postprocessed_decision_tree(X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None):

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
            random_state=random_seed + i,
            class_weight=class_weights  # Use the externally provided class weights
        )
        clf.fit(X_train, y_train)

        # Post-process pruning using cost complexity pruning
        path = clf.cost_complexity_pruning_path(X_train, y_train)
        ccp_alphas = path.ccp_alphas
        pruned_clf = None
        best_pruned_score = 0

        for ccp_alpha in ccp_alphas:
            # 确保 ccp_alpha 为非负值
            if ccp_alpha < 0:
                ccp_alpha = 0.0

            temp_clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                ccp_alpha=ccp_alpha,
                random_state=random_seed + i,
                class_weight=class_weights
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

    # Visualize the best Pruned Decision Tree and output IF-THEN rules to a file
    dot_data = export_graphviz(
        best_model, feature_names=X.columns, class_names=best_model.classes_.astype(str),
        filled=True, rounded=True, special_characters=True
    )
    graph = graphviz.Source(dot_data)
    graph.render(f"{output_dir}/best_pruned_postprocessed_tree", format='png', cleanup=True)

    print("================================================================================")
    print(f"Best Model Parameters (Pruned and Post-processed): {best_params}")
    print(f"Best Model Test Accuracy (Pruned and Post-processed): {best_accuracy:.4f}")

    # Output IF-THEN rules to a file
    ifthen_rules = export_text(best_model, feature_names=list(X.columns))
    with open(f"{output_dir}/best_pruned_postprocessed_tree_rules.txt", "w") as file:
        file.write("IF-THEN rules for the best pruned and post-processed decision tree:\n")
        file.write(ifthen_rules)
    print("IF-THEN rules for the best pruned and post-processed decision tree have been saved to file.")

    # Feature Importance
    feature_importances = best_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    feature_importance_df.to_csv(f"{output_dir}/best_pruned_postprocessed_tree_feature_importance.csv", index=False)
    print("Feature importance for the best pruned and post-processed decision tree has been saved to CSV.")

    # Compute summary statistics
    metrics_df = pd.DataFrame(experiment_metrics)
    summary_statistics = metrics_df[['Test Accuracy', 'F1 Score', 'AUC']].agg(['mean', 'var', 'std']).T

    # Return metrics, summary statistics, best model, best parameters, best scores, and feature importances
    return metrics_df, summary_statistics, best_model, best_params, best_accuracy, best_auc, best_f1


param_spaces = {
    "Random Forest": {
        'n_estimators': [100, 200, 300],  # 增加树的数量
        'max_depth': [10, 20, 30],  # 加深树的深度
        'class_weight': ['balanced', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]  # 控制每棵树的特征数
    },
    "XGBoost": {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'eval_metric': ['auc'],
        'subsample': [0.8, 0.9, 1.0],  # 每棵树的样本比例
        'colsample_bytree': [0.8, 0.9, 1.0]  # 每棵树的特征比例
    },
    "Gradient Boosting": {
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'subsample': [0.8, 0.9, 1.0],  # 样本比例
        'max_features': ['sqrt', 'log2', None]  # 特征数量
    },
    "Neural Network (Adam)": {
        'hidden_layer_sizes': [(50,), (100,), (150,)],  # 增加隐藏层规模
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [300, 500, 700],  # 增加迭代次数
        'batch_size': [32, 64, 128]  # 批量大小
    },
    "Neural Network (SGD)": {
        'hidden_layer_sizes': [(50,), (100,), (150,)],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.001, 0.01, 0.05],  # 微调学习率
        'max_iter': [300, 500, 700],
        'momentum': [0.8, 0.9, 0.95],  # 加入更小的动量选项
        'n_iter_no_change': [5, 10, 20],  # 提前停止
        'batch_size': [32, 64, 128]
    }
}


def random_search_and_evaluate_metrics(model_name, model_class, param_space, X, y, num_experiments, test_size, random_seed):
    best_accuracy = 0
    best_params = None
    best_model = None
    accuracies = []
    aucs = []
    f1_scores = []

    for i in range(num_experiments):
        # Randomly select hyperparameters from the search space
        params = list(ParameterSampler(param_space, n_iter=1, random_state=random_seed + i))[0]

        # Initialize the model with the selected hyperparameters
        model = model_class(**params)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

        # Compute evaluation metrics
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        auc = roc_auc_score(pd.get_dummies(y_test), y_pred_proba, multi_class='ovr') if y_pred_proba is not None else None

        # Store the metrics for each experiment
        accuracies.append(accuracy)
        f1_scores.append(f1)
        if auc is not None:
            aucs.append(auc)

        # Track the best model and parameters based on accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_params = params
            best_model = model

    # Calculate the mean, standard deviation, and variance for each metric
    mean_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    var_accuracy = np.var(accuracies)

    mean_auc = np.mean(aucs) if aucs else None
    std_auc = np.std(aucs) if aucs else None
    var_auc = np.var(aucs) if aucs else None

    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    var_f1 = np.var(f1_scores)

    # Output the best results and metrics statistics
    print("================================================================================")
    print(f"Best Hyperparameters for ({model_name}): {best_params}")
    print(f"Best Accuracy ({model_name}): {best_accuracy:.4f}")
    print(f"{model_name} - Mean Accuracy: {mean_accuracy:.4f}, Std: {std_accuracy:.4f}, Variance: {var_accuracy:.4f}")
    print(f"{model_name} - Mean AUC: {f'{mean_auc:.4f}' if mean_auc is not None else 'N/A'}, "
          f"Std: {f'{std_auc:.4f}' if std_auc is not None else 'N/A'}, "
          f"Variance: {f'{var_auc:.4f}' if var_auc is not None else 'N/A'}")
    print(f"{model_name} - Mean F1: {mean_f1:.4f}, Std: {std_f1:.4f}, Variance: {var_f1:.4f}")


    return best_model, best_params, best_accuracy, mean_accuracy, std_accuracy, var_accuracy, mean_auc, std_auc, var_auc, mean_f1, std_f1, var_f1


def compare_l2_and_dropout(X, y, test_size, random_seed, output_dir):
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
    print("================================================================================")

    for i, params in enumerate(hyperparameter_combinations):
        dropout_rate = params["dropout_rate"]
        weight_decay = params["weight_decay"]

        # L2 Regularization Model (MLPClassifier with weight decay)
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

        # Dropout Model (TensorFlow with dropout layer)
        dropout_model = tf.keras.Sequential([
            layers.InputLayer(shape=(X_train.shape[1],)),
            layers.Dense(100, activation='relu'),
            layers.Dropout(dropout_rate),  # Apply Dropout
            layers.Dense(4, activation='softmax')
        ])
        dropout_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        dropout_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        y_pred_dropout = np.argmax(dropout_model.predict(X_test), axis=1)
        dropout_accuracy = accuracy_score(y_test, y_pred_dropout)
        dropout_auc = roc_auc_score(pd.get_dummies(y_test), pd.get_dummies(y_pred_dropout), multi_class='ovr')
        dropout_f1 = f1_score(y_test, y_pred_dropout, average='weighted')

        # Store results for each combination
        results.append({
            "Combination": f"Dropout rate: {dropout_rate}, Weight decay (λ): {weight_decay}",
            "L2 Accuracy": l2_accuracy, "L2 AUC": l2_auc, "L2 F1": l2_f1,
            "Dropout Accuracy": dropout_accuracy, "Dropout AUC": dropout_auc, "Dropout F1": dropout_f1
        })


        print(f"Combination {i + 1}: Dropout rate {dropout_rate}, Weight decay {weight_decay}")
        print(f"L2 Model - Accuracy: {l2_accuracy:.4f}, AUC: {l2_auc:.4f}, F1 Score: {l2_f1:.4f}")
        print(f"Dropout Model - Accuracy: {dropout_accuracy:.4f}, AUC: {dropout_auc:.4f}, F1 Score: {dropout_f1:.4f}")

    # Convert to DataFrame and save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/l2_vs_dropout_results.csv")
    print("L2 vs Dropout comparison results saved to l2_vs_dropout_results.csv")


    return results_df





