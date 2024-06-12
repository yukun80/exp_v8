import shutil
import os

"""该程序主要作用是在多轮自训练之间扩充训练数据集和还原训练数据集
首先在自训练周期之间，将伪样本复制到训练数据集中，以扩充训练数据集
在自训练结束后，删除伪样本，还原训练数据集
"""


def copy_txt_files(source_dir, destination_dir):
    """强制复制路径中的文件
    Usage
    Replace 'source_directory' and 'destination_directory' with your actual paths
    """
    for filename in os.listdir(source_dir):
        source_file = os.path.join(source_dir, filename)
        destination_file = os.path.join(destination_dir, filename)

        if os.path.isfile(source_file):
            if os.path.exists(destination_file):
                print(f"File {filename} already exists. Replacing it.")
            shutil.copy2(source_file, destination_file)


def copy_tif_files(folder_path, txt_path, new_path):
    """复制图像文件到训练集"""
    # Open the txt file and read the filenames
    with open(txt_path, "r") as file:
        filenames = file.read().splitlines()

    # Iterate over the filenames
    for filename in filenames:
        # Find the file in the folder
        for root, dirs, files in os.walk(folder_path):
            expected_filename = f"{filename}.tif"
            if expected_filename in files:
                file_path = os.path.join(root, expected_filename)
                new_file_path = os.path.join(new_path, expected_filename)

                # Check if the file already exists in the new path
                if os.path.exists(new_file_path):
                    print(
                        f"File {expected_filename} already exists. Skipping..."
                    )
                else:
                    # Copy the file to the new path
                    shutil.copy2(file_path, new_file_path)
                    print(f"File {expected_filename} copied successfully.")
                break
        else:
            print(f"File {expected_filename} not found in the folder.")


def delete_files(txt_file_path, directory_path):
    """删除目录中的文件，文件名在txt文件中列出
    Usage
    delete_files("path_to_your_txt_file.txt", "path_to_your_directory")
    """
    # Open the txt file and read the file names
    with open(txt_file_path, "r") as file:
        file_names = file.read().splitlines()

    # Iterate over the file names
    for file_name in file_names:
        # Initialize a flag to check if a file has been deleted
        file_deleted = False
        # 在目录中搜索与任意扩展名的基本文件名匹配的文件
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                # Check if the file starts with the file_name and has an extension
                if os.path.splitext(file)[0] == file_name:
                    file_path = os.path.join(root, file)
                    # Delete the file
                    os.remove(file_path)
                    print(f"Deleted file: {file_path}")
                    file_deleted = True
        if not file_deleted:
            print(f"No matching file found for: {file_name}")


if __name__ == "__main__":
    """用伪样本扩充训练数据"""
    # pseudo_label_dir = r"results/CAS_yolo8_test/_yolo"
    # train_label_dir = r"datastes/yolo_dataset_sub/labels/train"

    # txt_dir = r"results/CAS_yolo8_test/result_list.txt"

    # test_tif_dir = r"datastes/CAS_yolo8_test"
    # train_tif_dir = r"datastes/yolo_dataset_sub/images/train"

    # # 扩充伪样本
    # copy_txt_files(pseudo_label_dir, train_label_dir)
    # # 扩充训练图像
    # copy_tif_files(test_tif_dir, txt_dir, train_tif_dir)

    # print("扩充数据集完成！")
    """删除所有伪样本，还原训练数据集"""
    pseudo_txt_dir = r"results/CAS_yolo8_test/result_list.txt"
    train_label_dir = r"datastes/yolo_dataset_sub/labels/train"
    train_img_dir = r"datastes/yolo_dataset_sub/images/train"

    delete_files(pseudo_txt_dir, train_label_dir)
    delete_files(pseudo_txt_dir, train_img_dir)
    print("删除伪样本完成！")
