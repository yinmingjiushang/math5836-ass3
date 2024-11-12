# math5836_ass3
group work by carol, hutton, mingyin

## run
environment:

python3 src/set_up.py

run:

way1: python3 src/main.py

way2: sh run.sh



## project structure
```bash
.
├── README.md                      # Project overview and instructions
├── data
│   ├── q_a
│   │   ├── q_a_data_X.csv         # Feature data for question A
│   │   └── q_a_data_y.csv         # Target data for question A
│   ├── q_b                         # Data for question B
│   └── q_c                         # Data for question C
├── flow_chart
│   ├── Boosting.dot               # Flowchart for boosting model
│   ├── L2_Dropout.dot             # Flowchart for model with L2 regularization and dropout
│   ├── NN.dot                     # Flowchart for neural network model
│   ├── PruneDT.dot                # Flowchart for pruned decision tree
│   ├── RF.dot                     # Flowchart for random forest model
│   ├── flow_chart.py              # Script to generate flowchart images
│   └── out                        # Output images of flowcharts
│       ├── Boosting.png
│       ├── L2_Dropout.png
│       ├── NN.png
│       ├── PruneDT.png
│       └── RF.png
├── full_directory_structure.txt    # Complete directory structure
├── out
│   ├── q_a
│   │   ├── analysis               # Performance analysis outputs for question A
│   │   │   ├── average_performance.png   # Average performance across experiments
│   │   │   ├── best_model_results.png    # Results of the best-performing model
│   │   │   └── stability.png             # Stability analysis of the model
│   │   └── visualizations         # Data exploration and feature analysis visualizations for question A
│   │       ├── correlation_heatmap.png   # Heatmap of feature correlations
│   │       ├── feature_distribution.png  # Distribution of features
│   │       ├── pairplot.png              # Pairplot showing relationships between variables
│   │       ├── summary_statistics.csv    # Summary statistics of the dataset
│   │       └── target_distribution.png   # Distribution of the target variable
│   ├── q_b                         # Analysis and visualizations for question B
│   └── q_c                         # Analysis and visualizations for question C
├── q_a_print_eg.txt                # Example output for question A
├── q_b_print_eg.txt                # Example output for question B
├── q_c_print_eg.txt                # Example output for question C
├── results
│   ├── q_a                         # Experiment results and metrics for question A
│   │   ├── basic_tree_prepruned_experiment_metrics.csv       # Metrics for pre-pruned decision tree experiments
│   │   ├── basic_tree_prepruned_summary_statistics.csv       # Summary statistics for pre-pruned decision tree
│   │   ├── best_basic_tree.png                              # Visualization of the best basic decision tree
│   │   ├── best_basic_tree_feature_importance.csv           # Feature importance for the best basic tree
│   │   ├── best_basic_tree_feature_importance.png           # Feature importance visualization for the best basic tree
│   │   ├── best_basic_tree_rules.txt                        # Rules for the best basic decision tree
│   │   ├── best_pruned_postprocessed_tree.png               # Visualization of the best post-processed pruned tree
│   │   ├── best_pruned_postprocessed_tree_feature_importance.csv  # Feature importance for pruned tree
│   │   ├── best_pruned_postprocessed_tree_feature_importance.png  # Feature importance visualization for pruned tree
│   │   ├── best_pruned_postprocessed_tree_rules.txt         # Rules for the pruned decision tree
│   │   ├── best_results.csv                                 # Summary of best results
│   │   ├── l2_vs_dropout_results.csv                        # Results comparing L2 and dropout
│   │   ├── model_sel.csv                                    # Model selection results
│   │   ├── postpruned_tree_experiment_metrics.csv           # Metrics for post-pruned decision tree experiments
│   │   ├── postpruned_tree_summary_statistics.csv           # Summary statistics for post-pruned decision tree
│   │   └── results.csv                                      # Comprehensive experiment results
│   ├── q_b                         # Experiment results and metrics for question B
│   └── q_c                         # Experiment results and metrics for question C
├── run.sh                          # Main script to execute the project
└── src
    ├── data_download.py            # Script for downloading data
    ├── data_input.py               # Data input module
    ├── data_processing.py          # Data preprocessing and cleaning
    ├── data_visualize.py           # Data visualization script
    ├── main.py                     # Main script for project execution
    ├── model_sel.py                # Model selection and evaluation code
    ├── models.py                   # Contains various model definitions
    ├── out_path.py                 # Configures output paths for results
    ├── q_a.py                      # Script specific to question A
    ├── q_b.py                      # Script specific to question B
    ├── q_c.py                      # Script specific to question C
    ├── requirements.txt            # Project dependencies
    ├── results_plot.py             # Results visualization
    ├── results_processing.py       # Processes results for statistical summaries
    └── set_up.py                   # Sets up the project directory structure
```