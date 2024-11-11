import  os
import shutil


def create_path():

    if os.path.exists("../out"):
        shutil.rmtree("../out")

    if os.path.exists("../results"):
        shutil.rmtree("../results")

    os.makedirs("../out", exist_ok=True)
    os.makedirs("../results", exist_ok=True)

    os.makedirs("../out/q_a/analysis", exist_ok=True)
    os.makedirs("../out/q_b/analysis", exist_ok=True)
    os.makedirs("../out/q_c/analysis", exist_ok=True)

    os.makedirs("../out/q_a/visualizations", exist_ok=True)
    os.makedirs("../out/q_b/visualizations", exist_ok=True)
    os.makedirs("../out/q_c/visualizations", exist_ok=True)

    os.makedirs("../results/q_a", exist_ok=True)
    os.makedirs("../results/q_b", exist_ok=True)
    os.makedirs("../results/q_c", exist_ok=True)

    os.makedirs("../data/q_a", exist_ok=True)
    os.makedirs("../data/q_b", exist_ok=True)
    os.makedirs("../data/q_c", exist_ok=True)



class OutPath:
    def X_path(flag):
        return f"../data/{flag}/{flag}_data_X.csv"

    def y_path(flag):
        return f"../data/{flag}/{flag}_data_y.csv"

    def results_path(flag):
        return f"../results/{flag}"

    def visualization_path(flag):
        return f"../out/{flag}/visualizations"

    def analysis_path(flag):
        return f"../out/{flag}/analysis"

    def results_store(flag):
        return f"../results/{flag}/results.csv"

    def best_results_store(flag):
        return f"../results/{flag}/best_results.csv"

