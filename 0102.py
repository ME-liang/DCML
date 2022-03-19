#!/usr/bin/env python
# -*- coding:utf-8 -*-

import time
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio
# from sklearn.externals import joblib
from joblib import *
import csv
import numpy as np
import os, glob
import scipy.io as sio
import xlwt
import xlrd

# import xlutils

# from xlutils.copy import copy

# def storFile(data, fileName):
#     with open(fileName, 'w', newline='') as f:
#         mywrite = csv.writer(f)
#         for d in data:
#             mywrite.writerow([d])

import time
from sklearn import metrics
import numpy as np
import pandas as pd
import scipy.io as sio
# from sklearn.externals import joblib
from joblib import *
import csv
import numpy as np
import os, glob
import scipy.io as sio
import xlwt
import xlrd
import xlsxwriter

# import xlutils

# from xlutils.copy import copy

# def storFile(data, fileName):
#     with open(fileName, 'w', newline='') as f:
#         mywrite = csv.writer(f)
#         for d in data:
#             mywrite.writerow([d])

# def storFile(data, a):
# num1 = np.ones(502)
# num1[:502] = 1
# # np.random.shuffle(num1)
# num2 = np.ones(169)
# num2[:169] = 0
# np.random.shuffle(num2)
# num3 = np.ones(765)
# num3[:765] = 2
# np.random.shuffle(num3)
# num4 = np.ones(859)
# num4[:859] = 3
# # np.random.shuffle(num4)
# num5 = np.ones(272)
# num5[:272] = 4
# # np.random.shuffle(num5)
# num6 = np.ones(485)
# num6[:485] = 5
# # np.random.shuffle(num6)
# num7 = np.ones(92)
# num7[:92] = 6
# # np.random.shuffle(num7)
# num8 = np.ones(410)
# num8[:410] = 7
# np.random.shuffle(num8)
import xlwt

workshh = xlwt.Workbook(encoding='utf-8')
worksheet = workshh.add_sheet('Sheet1', cell_overwrite_ok=True)
for i in range(1086):
    worksheet.write(i, 0, 0)
for i in range(1086, 1289):
    worksheet.write(i, 0, 1)
# for i in range(3724, 4037):
#     worksheet.write(i, 0, 1)
# for i in range(4037, 4344):
#     worksheet.write(i, 0, 0)
# for i in range(1088, 1853):
#     worksheet.write(i, 0, num3[i-1088])
# for i in range(1853, 2712):
#     worksheet.write(i, 0, num4[i-1853])
# for i in range(2712, 2984):
#     worksheet.write(i, 0, num5[i-2712])
# for i in range(2984, 3469):
#     worksheet.write(i, 0, num6[i-2984])
# for i in range(3469, 3561):
#     worksheet.write(i, 0, num7[i-3469])
# for i in range(3561, 3971):
#     worksheet.write(i, 0, num8[i-3561])

workshh.save('/DATA/LWN/new experiment2（复件）/Breat-Canner-Image-Classification2/COVID-CT-test.xls')
