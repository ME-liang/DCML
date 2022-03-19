import logging
import os
import re
from random import random


def remove_other_files(extension, archive_dir,source_dir):
    for root, dirs, files in os.walk(archive_dir):
        for current_file in files:

            if current_file.lower().endswith(extension):

                try:
                    logging.debug("Removing resource: File [%s]", os.path.join(root, current_file))
                    os.rename(
                        os.path.join(root, current_file),
                        os.path.join(source_dir, current_file))
                except OSError:
                    logging.error("Could not remove resource: File [%s]", os.path.join(root, current_file))


data_org_directory = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_v1.tar'  # raw data path
data_mdy_directory = '/home/cvnlp/LiChuanxiu/DataSets/BreaKHis_v1All'  # modified data stored path


remove_other_files(".png",data_org_directory,data_mdy_directory)