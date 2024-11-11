import subprocess
import sys

# List of required packages with specific versions
required_packages = [
    "graphviz==0.20.3",
    "imbalanced_learn==0.12.4",
    "matplotlib==3.9.2",
    "numpy==1.25.2",
    "pandas==2.2.3",
    "scikit_learn==1.5.2",
    "seaborn==0.13.2",
    "sympy==1.13.3",
    "tensorflow==2.17.0",
    "ucimlrepo==0.0.7",
    "xgboost==2.1.2"
]


def install_packages():
    for package in required_packages:
        package_name, version = package.split("==")
        try:
            # Check if the specific version is already installed
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True, text=True
            )

            if result.returncode == 0 and f"Version: {version}" in result.stdout:
                print(f"{package_name} {version} is already installed. Skipping installation.")
            else:
                # If not installed, or version mismatch, proceed with installation
                print(f"Installing {package}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", package])
                print(f"{package} installed successfully!")
        except subprocess.CalledProcessError as e:
            print(f"Error occurred while installing {package}: {e}")


if __name__ == "__main__":
    install_packages()
