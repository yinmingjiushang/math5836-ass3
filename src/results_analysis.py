import pandas as pd
import matplotlib.pyplot as plt

def best_model_results(file_path,out_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Plotting the line graph for different metrics
    plt.figure(figsize=(10, 6))

    # Defining distinct colors for each line
    colors = ['blue', 'green', 'red']

    # Plotting each metric as a separate line with specified colors
    for i, metric in enumerate(['Best Accuracy', 'Best AUC', 'Best F1 Score']):
        plt.plot(df['Model'], df[metric], label=metric, marker='o', color=colors[i])

    # Adding title and labels
    plt.title('Performance Metrics for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.legend()

    # Display the plot
    plt.tight_layout()
    plt.savefig(f"{out_path}/best_model_results.png")
    plt.close()



def plot_average_performance(df, output_path):
    # Plotting the average performance metrics
    plt.figure(figsize=(12, 6))

    # Plotting Mean Accuracy, Mean AUC, and Mean F1 Score
    for metric in ['Mean Accuracy', 'Mean AUC', 'Mean F1 Score']:
        plt.plot(df['Model'], df[metric], label=metric, marker='o')

    plt.title('Average Performance Metrics for Different Models')
    plt.xlabel('Model')
    plt.ylabel('Metric Value')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/average_performance.png")
    plt.close()

def plot_stability(df, output_path):
    # Plotting the stability of the models
    plt.figure(figsize=(12, 6))

    # Plotting Std Accuracy, Std AUC, and Std F1 Score
    for metric in ['Std Accuracy', 'Std AUC', 'Std F1 Score']:
        plt.plot(df['Model'], df[metric], label=metric, marker='o')

    plt.title('Stability of Different Models (Standard Deviation)')
    plt.xlabel('Model')
    plt.ylabel('Standard Deviation')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_path}/stability.png")
    plt.close()

def main():
    # Load the CSV file
    best_model_results("../results/q_a/best_model_results.csv","../out/q_a/model_results")


    file_path = '../results/q_a/model_results.csv'
    df = pd.read_csv(file_path)

    # Plot and save the figures
    plot_average_performance(df, "../out/q_a/model_results")
    plot_stability(df, "../out/q_a/model_results")

if __name__ == "__main__":
    main()





