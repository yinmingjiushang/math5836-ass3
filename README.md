# math5836_ass3



## project structure
```bash
.
├── README.md                     # Project overview and instructions
├── code_structure.txt            # Generated project directory structure file
├── data                          # Folder containing input data for different tasks
│   ├── q_a                       # Data for question/task 'q_a'
│   │   ├── q_a_data_X.csv        # Feature data for q_a
│   │   └── q_a_data_y.csv        # Label data for q_a
│   ├── q_b                       # Data for question/task 'q_b'
│   │   ├── q_b_data_X.csv        # Feature data for q_b
│   │   └── q_b_data_y.csv        # Label data for q_b
│   └── q_c                       # Data for question/task 'q_c'
│       ├── q_c_data_X.csv        # Feature data for q_c
│       └── q_c_data_y.csv        # Label data for q_c
├── out                           # Output folder with analysis and visualization results for each task
│   ├── q_a                       # Output results for q_a
│   │   ├── analysis              # Analysis results
│   │   │   ├── average_performance.png  # Average performance plot
│   │   │   ├── best_model_results.png   # Best model results plot
│   │   │   └── stability.png           # Stability analysis plot
│   │   └── visualizations        # Visualizations
│   │       ├── correlation_heatmap.png      # Correlation heatmap
│   │       ├── feature_distribution.png     # Feature distribution plot
│   │       ├── pairplot.png                 # Pairplot showing feature relationships
│   │       ├── summary_statistics.csv       # Summary statistics
│   │       └── target_distribution.png      # Target distribution plot
│   ├── q_b                       # Output results for q_b
│   │   ├── analysis              # Analysis results
│   │   │   ├── average_performance.png
│   │   │   ├── best_model_results.png
│   │   │   └── stability.png
│   │   └── visualizations        # Visualizations
│   │       ├── correlation_heatmap.png
│   │       ├── feature_distribution.png
│   │       ├── pairplot.png
│   │       ├── summary_statistics.csv
│   │       └── target_distribution.png
│   └── q_c                       # Output results for q_c
│       ├── analysis              # Analysis results
│       │   ├── average_performance.png
│       │   ├── best_model_results.png
│       │   └── stability.png
│       └── visualizations        # Visualizations
│           ├── correlation_heatmap.png
│           ├── feature_distribution.png
│           ├── pairplot.png
│           ├── summary_statistics.csv
│           └── target_distribution.png
├── results                       # Results folder containing model selection and evaluation results
│   ├── q_a                       # Results for q_a
│   │   ├── basic_tree_prepruned_experiment_metrics.csv    # Experiment metrics for basic unpruned tree
│   │   ├── basic_tree_prepruned_summary_statistics.csv    # Summary statistics for basic unpruned tree
│   │   ├── best_basic_tree.png                           # Best basic tree visualization
│   │   ├── best_basic_tree_feature_importance.csv        # Feature importance for the best basic tree
│   │   ├── best_basic_tree_feature_importance.png        # Feature importance plot
│   │   ├── best_basic_tree_rules.txt                     # Rules of the best basic tree
│   │   ├── best_pruned_postprocessed_tree.png            # Visualization of best pruned tree
│   │   ├── best_pruned_postprocessed_tree_feature_importance.csv
│   │   ├── best_pruned_postprocessed_tree_feature_importance.png
│   │   ├── best_pruned_postprocessed_tree_rules.txt
│   │   ├── best_results.csv                              # Best model results
│   │   ├── l2_vs_dropout_results.csv                     # L2 regularization vs Dropout results
│   │   ├── model_sel.csv                                 # Model selection results
│   │   ├── postpruned_tree_experiment_metrics.csv        # Experiment metrics for pruned tree
│   │   ├── postpruned_tree_summary_statistics.csv        # Summary statistics for pruned tree
│   │   └── results.csv                                   # Comprehensive results
│   ├── q_b                       # Results for q_b
│   │   ├── best_results.csv
│   │   ├── model_sel_results.csv
│   │   └── results.csv
│   └── q_c                       # Results for q_c
│       ├── best_results.csv
│       ├── model_sel_results.csv
│       └── results.csv
└── src                           # Source code folder containing various modules
    ├── data_download.py          # Data download module
    ├── data_input.py             # Data input module
    ├── data_processing.py        # Data processing module
    ├── data_visualize.py         # Data visualization module
    ├── main.py                   # Main entry point for the program
    ├── model_sel.py              # Model selection module
    ├── models.py                 # Model definitions
    ├── out_path.py               # Output path management module
    ├── results_plot.py           # Module for plotting results
    └── results_processing.py      # Results processing module
```