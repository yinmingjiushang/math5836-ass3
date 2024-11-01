import os
import shutil
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.neural_network import MLPClassifier
import graphviz
from tensorflow.keras import layers, models
import tensorflow as tf

# =========================
ed_state = 0
num_experiments = 5
test_size = 0.6
random_seed = 42
qa_1_path = "../out/q_a/visualizations"
qa_2_path = "../out/q_a/model_results"
# =========================

if ed_state == 0:
    import achieve_data
    if os.path.exists("../out"):
        shutil.rmtree("../out")
    os.makedirs("../out", exist_ok=True)
    os.makedirs(qa_1_path, exist_ok=True)
    os.makedirs(qa_2_path, exist_ok=True)


def data_cleaning(df):
    sex_mapping = {'M': 0, 'F': 1, 'I': 2}
    df['Sex'] = df['Sex'].map(sex_mapping)
    return df

def select_features_and_target(data):
    feature_columns = ['Sex', 'Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    X = data[feature_columns]
    y = data['Age Class']
    return X, y

def analyze_and_visualize_abalone_data(data, output_dir):
    age_bins = [0, 7, 10, 15, float('inf')]
    age_labels = ['Class 1: 0-7 years', 'Class 2: 8-10 years', 'Class 3: 11-15 years', 'Class 4: >15 years']
    data['Age Class'] = pd.cut(data['Rings'], bins=age_bins, labels=age_labels, right=True)
    feature_columns = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']

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

    visualize_age_class_distribution(data)
    visualize_feature_distribution(data, feature_columns)
    visualize_pairplot(data, feature_columns)
    visualize_correlation_heatmap(data, feature_columns)

def train_and_evaluate_model_with_experiments(model_class, model_params, data, num_experiments, test_size, random_seed):
    X, y = select_features_and_target(data)
    accuracies = []

    for i in range(num_experiments):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)
        model = model_class(**model_params, random_state=random_seed + i)
        model.fit(X_train, y_train)
        y_pred_test = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred_test))

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    return avg_accuracy, std_accuracy

def train_and_evaluate_unpruned_decision_tree(data, num_experiments, test_size, random_seed, output_dir):
    params_list = [{'max_depth': d, 'min_samples_split': s, 'min_samples_leaf': l}
                   for d in [None, 5, 10] for s in [2, 5] for l in [1, 2]]
    best_accuracy, best_model, best_params = 0, None, {}

    for params in params_list:
        avg_acc, std_acc = train_and_evaluate_model_with_experiments(DecisionTreeClassifier, params, data, num_experiments, test_size, random_seed)
        print(f"Params: {params}, Avg Accuracy: {avg_acc:.4f}, Std Accuracy: {std_acc:.4f}")
        if avg_acc > best_accuracy:
            best_accuracy, best_model, best_params = avg_acc, DecisionTreeClassifier(**params), params

    # Visualize the best Decision Tree
    X, y = select_features_and_target(data)
    best_model.fit(X, y)
    dot_data = export_graphviz(best_model, feature_names=X.columns, class_names=y.unique().astype(str),
                               filled=True, rounded=True, special_characters=True)
    graph = graphviz.Source(dot_data)
    graph.render(f"{output_dir}/best_decision_tree", format='png')
    return best_model, best_accuracy, best_params

def apply_random_forests(data, max_num_trees, num_experiments, test_size, random_seed, output_dir):
    X, y = select_features_and_target(data)
    accuracies = []

    for n_estimators in range(1, max_num_trees + 1):
        rf_accuracies = []
        for i in range(num_experiments):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)
            rf_model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_seed + i, n_jobs=-1)
            rf_model.fit(X_train, y_train)
            y_pred_test = rf_model.predict(X_test)
            rf_accuracies.append(accuracy_score(y_test, y_pred_test))

        avg_accuracy = np.mean(rf_accuracies)
        accuracies.append(avg_accuracy)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_num_trees + 1), accuracies, marker='o')
    plt.xlabel('Number of Trees')
    plt.ylabel('Average Accuracy')
    plt.title('Random Forest Accuracy by Tree Count')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/random_forest_accuracy.png")
    plt.close()

def train_and_evaluate_boosting_model(model_class, data, num_experiments, test_size, random_seed, output_dir):
    avg_acc, std_acc = train_and_evaluate_model_with_experiments(model_class, {}, data, num_experiments, test_size, random_seed)
    print(f"{model_class.__name__} Avg Accuracy: {avg_acc:.4f}, Std Accuracy: {std_acc:.4f}")

def train_and_evaluate_nn_optimizer(data, optimizer, num_experiments, test_size, random_seed):
    X, y = select_features_and_target(data)
    accuracies = []

    for i in range(num_experiments):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)
        nn_model = MLPClassifier(hidden_layer_sizes=(100,), solver=optimizer, max_iter=500, random_state=random_seed + i)
        nn_model.fit(X_train, y_train)
        y_pred_test = nn_model.predict(X_test)
        accuracies.append(accuracy_score(y_test, y_pred_test))

    avg_accuracy = np.mean(accuracies)
    std_accuracy = np.std(accuracies)
    print(f"Neural Network ({optimizer.upper()}) Avg Accuracy: {avg_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}")

def compare_l2_and_dropout(data, num_experiments, test_size, random_seed, output_dir):
    X, y = select_features_and_target(data)
    hyperparameter_combinations = [
        {'dropout_rate': 0.2, 'l2_lambda': 0.001},
        {'dropout_rate': 0.3, 'l2_lambda': 0.01},
        {'dropout_rate': 0.5, 'l2_lambda': 0.0001}
    ]

    for idx, params in enumerate(hyperparameter_combinations):
        accuracies = []
        for i in range(num_experiments):
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_seed + i)
            model = models.Sequential([
                layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda']), input_shape=(X_train.shape[1],)),
                layers.Dropout(params['dropout_rate']),
                layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(params['l2_lambda'])),
                layers.Dropout(params['dropout_rate']),
                layers.Dense(4, activation='softmax')
            ])
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), verbose=0)
            test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
            accuracies.append(test_accuracy)

        avg_accuracy = np.mean(accuracies)
        std_accuracy = np.std(accuracies)
        print(f"Model {idx + 1} - Dropout Rate: {params['dropout_rate']}, L2 Lambda: {params['l2_lambda']}")
        print(f"Avg Accuracy: {avg_accuracy:.4f}, Std Accuracy: {std_accuracy:.4f}\n")

def main():
    if ed_state == 0:
        achieve_data.achieve_data_main()

    abalone = pd.read_csv("../data/abalone_data.csv")
    abalone = data_cleaning(abalone)
    analyze_and_visualize_abalone_data(abalone, qa_1_path)

    print("\nTraining and Evaluating Unpruned Decision Tree:\n")
    train_and_evaluate_unpruned_decision_tree(abalone, num_experiments, test_size, random_seed, qa_2_path)

    print("\nApplying Random Forests:\n")
    apply_random_forests(abalone, max_num_trees=50, num_experiments=num_experiments, test_size=test_size, random_seed=random_seed, output_dir=qa_2_path)

    print("\nTraining and Evaluating Boosting Models:\n")
    train_and_evaluate_boosting_model(XGBClassifier, abalone, num_experiments, test_size, random_seed, qa_2_path)
    train_and_evaluate_boosting_model(GradientBoostingClassifier, abalone, num_experiments, test_size, random_seed, qa_2_path)

    print("\nTraining and Evaluating Neural Networks with Adam and SGD:\n")
    train_and_evaluate_nn_optimizer(abalone, 'adam', num_experiments, test_size, random_seed)
    train_and_evaluate_nn_optimizer(abalone, 'sgd', num_experiments, test_size, random_seed)

    print("\nComparing L2 Regularisation with Dropouts:\n")
    compare_l2_and_dropout(abalone, num_experiments, test_size, random_seed, qa_2_path)

if __name__ == "__main__":
    main()
