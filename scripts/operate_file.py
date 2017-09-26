# -*- coding:utf-8 -*-
import re
import os
import shutil
import time
import argparse

def split_literature(literature_path, doc_num_per_file, original_file, file_base_name):
    file_path_list = []
    for root, dirs, files in os.walk(literature_path):
        for file_name in files:
            file_path_list.append(os.path.join(root, file_name))
            # file_path_list.append(file_name)
    print file_path_list[:5]
    count_file = len(file_path_list) / doc_num_per_file
    for file_iter in range(count_file):
        new_file_path = os.path.join(original_file, file_base_name + str(file_iter))
        try:
            os.makedirs(new_file_path)
        except Exception:
            pass
        for doc_iter in range(file_iter * doc_num_per_file, (file_iter+1) * doc_num_per_file):
            shutil.move(file_path_list[doc_iter], new_file_path)
    new_file_path = os.path.join(original_file, file_base_name + "_other")
    try:
        os.makedirs(new_file_path)
    except Exception:
        pass
    for doc_iter in range(count_file * doc_num_per_file, len(file_path_list)):
        shutil.move(file_path_list[doc_iter], new_file_path)

if __name__ == '__main__':
    # shutil.move('E:\githubWorkSpace\mimic3-literatures-phenotyping\data-repository\DC_3D11.txt', 'E:\githubWorkSpace\mimic3-literatures-phenotyping\data-repository/new_literature')
    parser = argparse.ArgumentParser(description='splite the literatures into certain docs per file')
    parser.add_argument('literature_path', type=str, help='The path of literature')
    parser.add_argument('docs_num_per_file', type=int, help='The number of docs you want put in every file')
    parser.add_argument('splite_path', type=str, help='The path of split result')
    parser.add_argument('orginal_file_name', type=str, help='The name of split file')
    args, _ = parser.parse_known_args()

    split_literature(args.literature_path, args.docs_num_per_file, args.splite_path, args.original_file_name)
