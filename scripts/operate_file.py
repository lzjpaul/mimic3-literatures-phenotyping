# -*- coding:utf-8 -*-
import re
import os
import shutil
import time

def split_literature(literature_path, doc_num_per_file, original_file, file_base_name):
    file_path_list = []
    for root, dirs, files in os.walk(original_file):
        for file_name in files:
            file_path_list.append(os.path.join(root, file_name))
            # file_path_list.append(file_name)
    print file_path_list[:5]
    count_file = len(file_path_list) / doc_num_per_file
    other_file = len(file_path_list) - (count_file * doc_num_per_file)
    # for file_iter in range(count_file):


if __name__ == '__main__':
    shutil.move('E:\githubWorkSpace\mimic3-literatures-phenotyping\data-repository\DC_3D11.txt', 'E:\githubWorkSpace\mimic3-literatures-phenotyping\data-repository/new_literature')