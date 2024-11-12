import os
from graphviz import Source


def render_dot_files(dot_files, output_dir="./out", output_format="png"):
    """
    渲染多个 DOT 文件为图像格式并保存到当前文件夹。

    参数:
    - dot_files (list): 包含 DOT 文件路径的列表。
    - output_dir (str): 输出图像的目录（"." 表示当前文件夹）。
    - output_format (str): 输出图像的格式（默认 "png"）。
    """
    for dot_file in dot_files:
        # 提取文件名（不带路径和后缀）
        file_name = os.path.splitext(os.path.basename(dot_file))[0]

        # 加载 DOT 文件并渲染
        try:
            print(f"Rendering {dot_file}...")
            graph = Source.from_file(dot_file)
            output_path = os.path.join(output_dir, file_name)
            graph.render(filename=output_path, format=output_format, cleanup=True)
            print(f"Saved {output_path}")
        except Exception as e:
            print(f"Error rendering {dot_file}: {e}")


# 列出所有 DOT 文件
dot_files = ["Boosting.dot", "L2_Dropout.dot", "NN.dot", "PruneDT.dot", "RF.dot"]

# 调用函数生成图像，保存在当前文件夹
render_dot_files(dot_files)
