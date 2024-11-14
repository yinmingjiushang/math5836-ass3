# math5836_ass3
group work by carol, hutton, mingyin

## run
environment:
```bash
python3 src/set_up.py
```

run:
```bash
on local (Remove the comments within the main() function.):
python3 src/main.py

on ED:
sh run.sh

```

## Project Overview

### Part A: Abalone Age Prediction

1. **Dataset**  
   - **Objective**: Predict the age of abalone by classifying them into age groups based on ring counts.
   - **Age Groups**:
     - **Class 1:** 0-7 years
     - **Class 2:** 8-10 years
     - **Class 3:** 11-15 years
     - **Class 4:** Greater than 15 years  
   - **Source**: [Abalone Dataset](http://archive.ics.uci.edu/ml/datasets/Abalone)

2. **Data Analysis and Visualization**
   - Visualize the distribution of classes and features.
   - Generate additional plots to uncover insights and patterns.

3. **Modeling with Decision Tree**
   - Build a Decision Tree model for multiclass classification.
   - Perform multiple experiments (5 or more) with different hyperparameters (e.g., tree depth) to assess performance.
   - Select the best model and provide a visualization of the decision tree.
   - Translate selected nodes and leaves into "IF-THEN" rules for interpretability.
   - Explore pre-pruning and post-pruning techniques to improve model performance ([Scikit-Learn Pruning Example](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)).

4. **Random Forest Model**
   - Apply Random Forests and observe how increasing the number of trees affects performance.

5. **Comparison with XGBoost and Gradient Boosting**
   - Compare the performance of Decision Tree, Random Forest, XGBoost, and Gradient Boosting models.
   - Discuss results using relevant metrics such as accuracy, AUC, or F1 score.

6. **Comparison with Simple Neural Networks**
   - Implement neural networks using Adam or SGD optimizers with default hyperparameters.
   - Compare their performance with the tree-based models.
   - Experiment with different configurations of L2 regularization (weight decay) and dropout rates using Adam optimizer.
   - Present results for at least 3 different hyperparameter combinations.

### Part B: Contraceptive Method Choice Classification

1. **Dataset**  
   - **Objective**: Classify contraceptive method choices based on survey data.
   - **Details**:
     - **Number of Instances**: 1,473
     - **Number of Attributes**: 10 (including the class attribute)
   - **Source**: [Contraceptive Method Choice Dataset](https://archive.ics.uci.edu/dataset/30/contraceptive+method+choice)

2. **Data Preprocessing and Visualization**
   - Clean the dataset by handling special characters (e.g., `?` and `$`).
   - Visualize data distributions and identify any patterns.

3. **Model Application**
   - Apply the two best-performing models from Part A to this dataset.
   - Report results using appropriate metrics such as F1 score or ROC-AUC.

### Part C: Additional Dataset Analysis

1. **Dataset**  
   - **Objective**: Choose and analyze a dataset from the UCI Machine Learning Repository.
   - **Task**: Perform multiclass classification or regression, depending on the dataset selected.

2. **Visualization and Model Building**
   - Conduct data visualization to understand the dataset.
   - Build suitable models and report error metrics for both training and test sets.

### Additional Task: Addressing Class Imbalance

- **SMOTE**  
  - Utilize SMOTE (Synthetic Minority Over-sampling Technique) to address class imbalance in any of the datasets.
  - Analyze and report the impact on model performance.


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

## notice
1. Running the three datasets on ed results in a “kill” situation, mostly occurring during part c. Apologies for not being able to fully debug it on ed.
2. I recorded the terminal output and logs from local runs in `results_local_eg` as a reference.


