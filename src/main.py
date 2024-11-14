import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

import data_input,data_processing,data_visualize,model_sel,models,out_path,results_plot,results_processing



# define
# =========================
ed_state = 0     # 0 == local  ,  1 == ed
num_experiments = 30
# model train param
test_size = 0.6
random_seed = 42
# =========================
if ed_state == 0:
    import data_download


def data_and_visualize(flag):

    # data download
    if ed_state == 0:
        data_download.data_download(flag)

    # data input
    X, y = data_input.data_input(flag)

    # data processing
    X, y = data_processing.data_clean(X, y)

    if flag == "q_a":
        y = data_processing.map_values_to_bins(flag, y)

    data_visualize.visualize_data(flag, X, y)

    y = LabelEncoder().fit_transform(y)

    # handle_imbalance
    X, y, class_weights = data_processing.handle_imbalance(X, y, "weighted")

    return X, y, class_weights


def question_a():

    results = []
    best_results = []

    flag = "q_a"
    X, y, class_weights = data_and_visualize(flag)

    # basic decision tree
    metrics_df_basic, summary_stats_basic, best_model_basic, best_params_basic, best_accuracy_basic, best_auc_basic, best_f1_basic = models.train_and_evaluate_basic_decision_tree(
        X, y, num_experiments, test_size, random_seed, out_path.OutPath.results_path(flag), class_weights
    )
    results, best_results = results_processing.save_tree_results(metrics_df_basic, summary_stats_basic, out_path.OutPath.results_path(flag), results, best_results, best_accuracy_basic, best_auc_basic, best_f1_basic, "basic_tree_prepruned")

    # postpruned decision tree
    metrics_df_pruned, summary_stats_pruned, best_model_pruned, best_params_pruned, best_accuracy_pruned, best_auc_pruned, best_f1_pruned = models.train_and_evaluate_pruned_postprocessed_decision_tree(
        X, y, num_experiments, test_size, random_seed, out_path.OutPath.results_path(flag), class_weights
    )
    results, best_results = results_processing.save_tree_results(metrics_df_pruned, summary_stats_pruned, out_path.OutPath.results_path(flag), results, best_results, best_accuracy_pruned, best_auc_pruned, best_f1_pruned, "postpruned_tree")

    # Perform random hyperparameter search and evaluation for each model
    for model_name, model_class in [
        ("Random Forest", RandomForestClassifier),
        ("XGBoost", XGBClassifier),
        ("Gradient Boosting", GradientBoostingClassifier),
        ("Neural Network (Adam)", MLPClassifier),
        ("Neural Network (SGD)", MLPClassifier)
    ]:
        param_space = models.param_spaces[model_name]
        best_model, best_params, best_accuracy, best_auc, best_f1, mean_accuracy, std_accuracy, var_accuracy, mean_auc, std_auc, var_auc, mean_f1, std_f1, var_f1 = models.random_search_and_evaluate_metrics(
            model_name, model_class, param_space, X, y, num_experiments, test_size, random_seed
        )

        # Append mean statistics
        results = results_processing.append_mean_statistics(results, model_name, mean_accuracy, var_accuracy, std_accuracy, mean_auc, var_auc, std_auc, mean_f1, var_f1, std_f1)

        # Append best results
        best_results = results_processing.append_best_results(best_results, model_name, best_accuracy, best_auc, best_f1)

    # l2 vs dropout
    best_params, best_accuracy, best_auc, best_f1, mean_accuracy, std_accuracy, var_accuracy, mean_auc, std_auc, var_auc, mean_f1, std_f1, var_f1 = models.compare_l2_and_dropout(X, y,
                                                                                                                                                                                  num_experiments,
                                                                                                                                                                                  test_size,
                                                                                                                                                                                  random_seed,
                                                                                                                                                                                  out_path.OutPath.results_path(
                                                                                                                                                                                      flag))
    # Append mean statistics
    results = results_processing.append_mean_statistics(results, "L2 vs Dropout", mean_accuracy, var_accuracy, std_accuracy, mean_auc, var_auc, std_auc, mean_f1, var_f1, std_f1)

    # Append best results
    best_results = results_processing.append_best_results(best_results, "L2 vs Dropout", best_accuracy, best_auc, best_f1)

    results_processing.save_results_to_csv(results, out_path.OutPath.results_store(flag))
    results_processing.save_results_to_csv(best_results, out_path.OutPath.best_results_store(flag))
    print("================================================================================")
    print(f"Mean results saved to {out_path.OutPath.results_store(flag)}")
    print(f"Best results saved to {out_path.OutPath.best_results_store(flag)}")

    results_plot.results_plot(flag)
    return 0


def question_b():

    flag = "q_b"

    results = []
    best_results = []

    X, y, class_weights = data_and_visualize(flag)

    models_name = model_sel.model_selection("q_a",2)

    sel_results = model_sel.models_sel_results(models_name, X, y, num_experiments, test_size, random_seed, out_path.OutPath.results_path(flag), class_weights)

    results, best_results = results_processing.process_model_sel_results(f"{out_path.OutPath.results_path(flag)}/model_sel_results.csv")

    results_processing.save_results_to_csv(results, out_path.OutPath.results_store(flag))
    results_processing.save_results_to_csv(best_results, out_path.OutPath.best_results_store(flag))
    print("================================================================================")
    print(f"Mean results saved to {out_path.OutPath.results_store(flag)}")
    print(f"Best results saved to {out_path.OutPath.best_results_store(flag)}")

    results_plot.results_plot(flag)

    return 0

def question_c():
    flag = "q_c"

    results = []
    best_results = []

    X, y, class_weights = data_and_visualize(flag)

    models_name = model_sel.model_selection("q_a", 2)

    sel_results = model_sel.models_sel_results(models_name, X, y, num_experiments, test_size, random_seed, out_path.OutPath.results_path(flag), class_weights)

    results, best_results = results_processing.process_model_sel_results(f"{out_path.OutPath.results_path(flag)}/model_sel_results.csv")

    results_processing.save_results_to_csv(results, out_path.OutPath.results_store(flag))
    results_processing.save_results_to_csv(best_results, out_path.OutPath.best_results_store(flag))
    print("================================================================================")
    print(f"Mean results saved to {out_path.OutPath.results_store(flag)}")
    print(f"Best results saved to {out_path.OutPath.best_results_store(flag)}")

    results_plot.results_plot(flag)

    return 0


def main():

    # mkdir path
    if ed_state == 0:
        out_path.create_path()

    # # q_a
    # print("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
    # question_a()
    #
    # # q_b
    # print("bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb")
    # question_b()
    #
    # # q_c
    # print("cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc")
    # question_c()

if __name__ == "__main__":
        main()