import logging
import os
import re
import shutil

# def label_files(directory, category_rules, btarDir, mtarDir):
#
#     for root, dirs, files in os.walk(directory):
#         for file in files:
#
#             if re.compile(list(category_rules.values())[0]).match(file):
#                 shutil.copyfile(os.path.join(root,file), btarDir + file)
#
#
#
#             elif re.compile(list(category_rules.values())[1]).match(file):
#                 shutil.copyfile(os.path.join(root,file), mtarDir + file)
#
#
# data_org_directory = '/home/cvnlp/LiChuanxiu/Augment/60'  # raw data path
# btarDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/40/predict/benign/'
# mtarDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/40/train/malignant/'
#
#
# #'SOB_B_.*.png'
# label_files(data_org_directory, {'benign': 'SOB_B_.*.-40-.*.png', 'malignant': 'SOB_M_.*.-40-.*.png'}, btarDir, mtarDir)


def label_files_multi(directory, category_rules, ADir, FDir, TADir, PTDir, DCDir, LCDir, MCDir, PCDir):

    for root, dirs, files in os.walk(directory):
        for file in files:


            if re.compile(list(category_rules.values())[0]).match(file):
                shutil.copyfile(os.path.join(root,file), ADir + file)
            elif re.compile(list(category_rules.values())[1]).match(file):
                shutil.copyfile(os.path.join(root,file), FDir + file)
            elif re.compile(list(category_rules.values())[2]).match(file):
                shutil.copyfile(os.path.join(root, file), PTDir + file)
            elif re.compile(list(category_rules.values())[3]).match(file):
                shutil.copyfile(os.path.join(root, file), TADir + file)
            elif re.compile(list(category_rules.values())[4]).match(file):
                shutil.copyfile(os.path.join(root, file), DCDir + file)
            elif re.compile(list(category_rules.values())[5]).match(file):
                shutil.copyfile(os.path.join(root, file), LCDir + file)
            elif re.compile(list(category_rules.values())[6]).match(file):
                shutil.copyfile(os.path.join(root, file), MCDir + file)
            elif re.compile(list(category_rules.values())[7]).match(file):
                shutil.copyfile(os.path.join(root, file), PCDir + file)


data_org_directory = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis/400/train'

ADir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/A/'
FDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/F/'
TADir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/TA/'
PTDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/PT/'
DCDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/DC/'
LCDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/LC/'
MCDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/MC/'
PCDir = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_Multi/400/train/PC/'


a = {'A': 'SOB_B_A.*.-400-.*.png', 'F': 'SOB_B_F.*.-400-.*.png', 'PT': 'SOB_B_PT.*.-400-.*.png', 'TA': 'SOB_B_TA.*.-400-.*.png', 'DC': 'SOB_M_DC.*.-400-.*.png', 'LC': 'SOB_M_LC.*.-400-.*.png', 'MC': 'SOB_M_MC.*.-400-.*.png', 'PC': 'SOB_M_PC.*.-400-.*.png'}

label_files_multi(data_org_directory,a,ADir, FDir, TADir, PTDir, DCDir, LCDir, MCDir, PCDir)
