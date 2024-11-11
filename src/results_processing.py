import pandas as pd

import out_path

def calculate_and_append_summary(summary_statistics, results, model_name):

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
    return results

def save_tree_results(metrics_df, summary, out_path, results, best_results, best_accuracy, best_auc, best_f1, model_name):

    # Save metrics and summary to CSV
    metrics_df.to_csv(f"{out_path}/{model_name}_experiment_metrics.csv", index=False)
    summary.to_csv(f"{out_path}/{model_name}_summary_statistics.csv", index=True)

    # Calculate and append summary statistics
    results = calculate_and_append_summary(summary, results, model_name)

    # Append best results to best_results list
    best_results.append({
        "Model": model_name,
        "Best Accuracy": best_accuracy,
        "Best AUC": best_auc,
        "Best F1 Score": best_f1
    })
    return results, best_results

# Function to append mean statistics to results list
def append_mean_statistics(results, model_name, mean_accuracy, var_accuracy, std_accuracy,
                           mean_auc, var_auc, std_auc, mean_f1, var_f1, std_f1):
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

    return results

# Function to append best results to best_results list
def append_best_results(best_results, model_name, best_accuracy, best_auc, best_f1):
    best_results.append({
        "Model": model_name,
        "Best Accuracy": best_accuracy,
        "Best AUC": best_auc,
        "Best F1 Score": best_f1
    })

    return best_results

def l2_best_results(best_results, results_df):

    # Append L2 vs Dropout best results for summary
    l2_best_accuracy = max(results_df["L2 Accuracy"].max(), results_df["Dropout Accuracy"].max())
    l2_best_auc = max(results_df["L2 AUC"].max(), results_df["Dropout AUC"].max())
    l2_best_f1 = max(results_df["L2 F1"].max(), results_df["Dropout F1"].max())

    best_results.append({
        "Model": "L2 vs Dropout",
        "Best Accuracy": l2_best_accuracy,
        "Best AUC": l2_best_auc,
        "Best F1 Score": l2_best_f1
    })

    return best_results

def process_model_sel_results(csv_path):
    # 加载 CSV 文件
    model_results_df = pd.read_csv(csv_path)

    # 列表存储处理后的结果
    results = []
    best_results = []

    # 遍历 CSV 的每一行
    for _, row in model_results_df.iterrows():
        # 将平均统计附加到 results 列表
        append_mean_statistics(
            results,
            model_name=row.get('Model', None),
            mean_accuracy=row.get('Mean Accuracy', None),
            var_accuracy=row.get('Var Accuracy', None),
            std_accuracy=row.get('Std Accuracy', None),
            mean_auc=row.get('Mean AUC', None),
            var_auc=row.get('Var AUC', None),
            std_auc=row.get('Std AUC', None),
            mean_f1=row.get('Mean F1', None),
            var_f1=row.get('Var F1', None),
            std_f1=row.get('Std F1', None)
        )

        # 将最佳结果附加到 best_results 列表
        append_best_results(
            best_results,
            model_name=row.get('Model', None),
            best_accuracy=row.get('Best Accuracy', None),
            best_auc=row.get('Best AUC', None),
            best_f1=row.get('Best F1', None)
        )

    # 转换为 DataFrame
    results_df = pd.DataFrame(results)
    best_results_df = pd.DataFrame(best_results)

    return results_df, best_results_df

def save_results_to_csv(results, output_file):
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)