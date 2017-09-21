import os
import argparse

class Directory(object):

    @staticmethod
    def folder_process(folder_path):
        file_path_list = []
        for root, dirs, files in os.walk(folder_path):
            for file_name in files:
                file_path_list.append(os.path.join(root, file_name))
                # file_path_list.append(file_name)
        return file_path_list


# test
# print os.path.join('../data-repository/BMC_Musculoskelet_Disord')
# print "/Users/yangkai/PycharmProjects/github-repository/mimic3-literatures-phenotyping/data-repository/BMC_Musculoskelet_Disord"
# print len(Directory.folder_process(os.path.join(os.path.split(os.path.abspath(os.path.dirname(__file__)))[0]
#                                                 ,'data-repository/BMC_Musuloskelet_Disord')))
# files = Directory.folder_process("E:\medical data\PubMed_Publications\comm_use.0-9A-B")
# for file in files:
#     print file
#     print os.path.basename(file)
# print len(files)
# print len(set(files))
# parser = argparse.ArgumentParser(description="Extract the knowledge from literature dataset.")
# parser.add_argument('doc_origin_path', type=str,
#                         help='Original literature directory may containing several folders.')
# args, _ = parser.parse_known_args()
# print 'the document size :', len(Directory.folder_process(args.doc_origin_path))