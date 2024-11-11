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


# model_mapping = {
#     "basic_tree_prepruned": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.train_and_evaluate_basic_decision_tree(
#         X, y, num_experiments, test_size, random_seed, output_dir, class_weights),
#     "postpruned_tree": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.train_and_evaluate_pruned_postprocessed_decision_tree(
#         X, y, num_experiments, test_size, random_seed, output_dir, class_weights),
#     "Random Forest": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
#         "Random Forest", models.RandomForestClassifier, models.param_spaces["Random Forest"], X, y, num_experiments, test_size, random_seed),
#     "XGBoost": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
#         "XGBoost", models.XGBClassifier, models.param_spaces["XGBoost"], X, y, num_experiments, test_size, random_seed),
#     "Gradient Boosting": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
#         "Gradient Boosting", models.GradientBoostingClassifier, models.param_spaces["Gradient Boosting"], X, y, num_experiments, test_size, random_seed),
#     "Neural Network (Adam)": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
#         "Neural Network (Adam)", models.MLPClassifier, models.param_spaces["Neural Network (Adam)"], X, y, num_experiments, test_size, random_seed),
#     "Neural Network (SGD)": lambda X, y, num_experiments, test_size, random_seed, output_dir, class_weights=None: models.random_search_and_evaluate_metrics(
#         "Neural Network (SGD)", models.MLPClassifier, models.param_spaces["Neural Network (SGD)"], X, y, num_experiments, test_size, random_seed),
#     "L2 vs Dropout": lambda X, y, test_size, random_seed, output_dir: models.compare_l2_and_dropout(X, y, test_size, random_seed, output_dir)
# }

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
    "L2 vs Dropout": lambda X, y, num_experiments, test_size, random_seed, output_dir: models.compare_l2_and_dropout(
        X, y, num_experiments, test_size, random_seed, output_dir)}



def models_sel_results(model_names, X, y, num_experiments, test_size, random_seed, output_dir, class_weights):
    results_summary = []
    num_experiments = int(num_experiments)

    for model_name in model_names:
        # 检查模型名称并传入对应参数
        if model_name == "L2 vs Dropout":
            # 获取 L2 vs Dropout 的返回值
            (best_params, best_accuracy, best_auc, best_f1,
             mean_accuracy, std_accuracy, var_accuracy,
             mean_auc, std_auc, var_auc,
             mean_f1, std_f1, var_f1) = model_mapping[model_name](X, y, num_experiments, test_size, random_seed, output_dir)

            # 保存结果到 CSV 文件
            print(f"L2 vs Dropout results saved to {output_dir}/l2_vs_dropout_results.csv")

            # 添加 L2 vs Dropout 的结果到 summary
            result = {
                "Model": model_name,
                "Best Parameters": best_params,
                "Best Accuracy": best_accuracy,
                "Best AUC": best_auc,
                "Best F1": best_f1,
                "Mean Accuracy": mean_accuracy,
                "Std Accuracy": std_accuracy,
                "Var Accuracy": var_accuracy,
                "Mean AUC": mean_auc,
                "Std AUC": std_auc,
                "Var AUC": var_auc,
                "Mean F1": mean_f1,
                "Std F1": std_f1,
                "Var F1": var_f1
            }

        elif model_name in ["basic_tree_prepruned", "postpruned_tree"]:
            # 需要 num_experiments 和 class_weights
            result_tuple = model_mapping[model_name](X, y, num_experiments, test_size, random_seed, output_dir, class_weights)

            # 根据返回值解包
            metrics_df, summary_statistics, best_model, best_params, best_accuracy, best_auc, best_f1 = result_tuple
            result = {
                "Model": model_name,
                "Best Parameters": best_params,
                "Best Accuracy": best_accuracy,
                "Best AUC": best_auc,
                "Best F1": best_f1
            }

        else:
            # 其他模型只需要 test_size 和 class_weights
            result_tuple = model_mapping[model_name](X, y, num_experiments, test_size, random_seed, output_dir)

            # 随机搜索的返回值，包含最佳 AUC 和最佳 F1
            (best_model, best_params, best_accuracy, best_auc, best_f1,
             mean_accuracy, std_accuracy, var_accuracy,
             mean_auc, std_auc, var_auc, mean_f1, std_f1, var_f1) = result_tuple

            result = {
                "Model": model_name,
                "Best Parameters": best_params,
                "Best Accuracy": best_accuracy,
                "Best AUC": best_auc,
                "Best F1": best_f1,
                "Mean Accuracy": mean_accuracy,
                "Std Accuracy": std_accuracy,
                "Var Accuracy": var_accuracy,
                "Mean AUC": mean_auc,
                "Std AUC": std_auc,
                "Var AUC": var_auc,
                "Mean F1": mean_f1,
                "Std F1": std_f1,
                "Var F1": var_f1
            }

        # 将每个模型的结果添加到总 summary 中
        results_summary.append(result)

    # 将结果保存到 CSV 文件
    results_df = pd.DataFrame(results_summary)
    output_file = f"{output_dir}/model_sel_results.csv"
    results_df.to_csv(output_file, index=False)
    print(f"Complete results have been saved to {output_file}")

    return results_summary
