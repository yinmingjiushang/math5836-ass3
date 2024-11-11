import pandas as pd

import out_path
import models


def model_selection(flag, number):
    # 读取 CSV 文件
    df = pd.read_csv(out_path.OutPath.best_results_store(flag))

    # 限制 number 不超过数据行数上限
    number = min(number, len(df))

    # 根据 'Best Accuracy' 排序并选择前 number 个模型
    top_models = df.sort_values(by='Best Accuracy', ascending=False).head(number)
    print(top_models)

    # 保存前 number 个模型信息到新的 CSV 文件
    top_models.to_csv(f'{out_path.OutPath.results_path(flag)}/model_sel.csv', index=False)

    # 获取前 number 个模型的名称
    model_names = top_models['Model'].tolist()

    return model_names


model_mapping = {
    "basic_tree_prepruned": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.train_and_evaluate_basic_decision_tree(
        X, y, num_experiments, test_size, random_seed, output_dir, class_weights),
    "postpruned_tree": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.train_and_evaluate_pruned_postprocessed_decision_tree(
        X, y, num_experiments, test_size, random_seed, output_dir, class_weights),
    "Random Forest": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
        "Random Forest", models.RandomForestClassifier, models.param_spaces["Random Forest"], X, y, num_experiments, test_size, random_seed),
    "XGBoost": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
        "XGBoost", models.XGBClassifier, models.param_spaces["XGBoost"], X, y, num_experiments, test_size, random_seed),
    "Gradient Boosting": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
        "Gradient Boosting", models.GradientBoostingClassifier, models.param_spaces["Gradient Boosting"], X, y, num_experiments, test_size, random_seed),
    "Neural Network (Adam)": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
        "Neural Network (Adam)", models.MLPClassifier, models.param_spaces["Neural Network (Adam)"], X, y, num_experiments, test_size, random_seed),
    "Neural Network (SGD)": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
        "Neural Network (SGD)", models.MLPClassifier, models.param_spaces["Neural Network (SGD)"], X, y, num_experiments, test_size, random_seed),
    "L2 vs Dropout": lambda X, y, test_size, random_seed, output_dir: models.compare_l2_and_dropout(X, y, test_size, random_seed, output_dir)
}


def models_sel_results(model_names, X, y, num_experiments, test_size, random_seed, path, class_weights):
    results_summary = []

    for model_name in model_names:
        # 调用 model_mapping 中的函数并解包返回的元组
        result_tuple = model_mapping[model_name](X, y, num_experiments, test_size, random_seed, path, class_weights)

        # 将元组结果解包并手动构建字典
        result = {
            "Model": model_name,
            "best_model": result_tuple[0],
            "best_params": result_tuple[1],
            "best_accuracy": result_tuple[2],
            "mean_accuracy": result_tuple[3],
            "std_accuracy": result_tuple[4],
            "var_accuracy": result_tuple[5],
            "mean_auc": result_tuple[6],
            "std_auc": result_tuple[7],
            "var_auc": result_tuple[8],
            "mean_f1": result_tuple[9],
            "std_f1": result_tuple[10],
            "var_f1": result_tuple[11],
            "feature_importances": result_tuple[12] if len(result_tuple) > 12 else "N/A"
        }

        # 打印或存储每个模型的详细摘要
        print("================================================================================")
        print(f"sel_Model: {model_name}")
        print(f"Best Accuracy: {result['best_accuracy']}")
        print(f"Best Parameters: {result['best_params']}")
        print(f"Mean Accuracy: {result['mean_accuracy']}, Std: {result['std_accuracy']}, Var: {result['var_accuracy']}")
        print(f"Mean AUC: {result['mean_auc']}, Std: {result['std_auc']}, Var: {result['var_auc']}")
        print(f"Mean F1: {result['mean_f1']}, Std: {result['std_f1']}, Var: {result['var_f1']}")
        if result['feature_importances'] != "N/A":
            print("Feature Importances saved.")

        # 将完整结果添加到 results_summary 列表
        results_summary.append(result)

    # 将结果转换为 DataFrame 并保存为 CSV 文件
    results_df = pd.DataFrame(results_summary)
    output_file = f"{path}/model_sel_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Complete results have been saved to {output_file}")

    return results_summary


