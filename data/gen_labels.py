# import pickle
#
#
# def convert_txt_to_pkl(txt_file, pkl_file):
#     data = {}
#     with open(txt_file, 'r') as f:
#         for line in f:
#             key, value = line.strip().split('\t')
#             data[key] = value
#     with open(pkl_file, 'wb') as f:
#         pickle.dump(data, f)
#
# convert_txt_to_pkl('train_caption.txt', 'labels.pkl')

import pickle
import os

pkl_file = "train_labels.pkl"


def txt_to_pkl(dir_path):
    # 创建一个字典来存储文件名和内容
    data = {}

    # 遍历目录中的所有文件
    for filename in os.listdir(dir_path):
        # 只读取.txt文件
        if filename.endswith(".txt"):
            file_path = os.path.join(dir_path, filename)
            with open(file_path, "r") as f:
                # 将文件名作为键，文件内容作为值存入字典
                filename = filename[:-4]
                data[filename] = f.read()

    # 将字典写入pkl文件
    mode = "wb"
    with open(pkl_file, mode) as f:
        pickle.dump(data, f)


# 调用函数，将目录中的txt文件写入pkl文件
txt_to_pkl("train_hyb")

