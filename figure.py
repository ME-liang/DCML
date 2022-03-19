import xlrd
import matplotlib.pyplot as plt

ExcelFile = xlrd.open_workbook('/home/cvnlp/test_100.xlsx')
ExcelFile1 = xlrd.open_workbook('/home/cvnlp/test_100_limit.xlsx')

# sheet = ExcelFile.sheet_by_name('epoch_trainloss')
# epoch_trainloss = sheet.col_values(0)
# sheet = ExcelFile.sheet_by_name('epoch_testloss')
# epoch_testloss = sheet.col_values(0)
#
#
# plt.xlabel("Training Epochs")
# plt.ylabel("Training Loss")
# plt.plot(range(1,len(epoch_trainloss)+1),epoch_trainloss,lw=2,label="epoch_trainloss")
# # plt.plot(range(1,len(epoch_testloss)+1),epoch_testloss,lw=2,label="epoch_testloss")
#
# plt.ylim((0,0.9))
# plt.xlim((0,len(epoch_trainloss)))
# plt.legend()
# plt.show()


# sheet = ExcelFile.sheet_by_name('epoch_trainacc')
# epoch_trainacc = sheet.col_values(0)
# sheet = ExcelFile.sheet_by_name('epoch_testacc')
# epoch_testacc = sheet.col_values(0)
#
# epoch_trainacc = list(map(lambda x:x*100, epoch_trainacc))
# epoch_testacc = list(map(lambda x:x*100, epoch_testacc))
# plt.xlabel("Training Epochs")
# plt.ylabel("Accuracy")
# plt.plot(range(1,len(epoch_trainacc)+1),epoch_trainacc,lw=2,label="epoch_trainacc")
# plt.plot(range(1,len(epoch_testacc)+1),epoch_testacc,lw=2,label="epoch_testacc")
#
# plt.ylim((50,100))
# plt.xlim((0,len(epoch_trainacc)))
# plt.legend()
# plt.show()

sheet = ExcelFile.sheet_by_name('epoch_trainloss')
epoch_trainloss = sheet.col_values(0)
sheet = ExcelFile1.sheet_by_name('epoch_trainloss')
epoch_testloss = sheet.col_values(0)


plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1,len(epoch_trainloss)+1),epoch_trainloss,lw=2,label="epoch_trainloss")
plt.plot(range(1,len(epoch_testloss)+1),epoch_testloss,lw=2,label="epoch_trainloss_limit")

plt.ylim((0,0.9))
plt.xlim((0,len(epoch_trainloss)))
plt.legend()
plt.show()


sheet = ExcelFile.sheet_by_name('epoch_trainacc')
epoch_trainacc = sheet.col_values(0)
sheet = ExcelFile1.sheet_by_name('epoch_trainacc')
epoch_testacc = sheet.col_values(0)

epoch_trainacc = list(map(lambda x:x*100, epoch_trainacc))
epoch_testacc = list(map(lambda x:x*100, epoch_testacc))
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,len(epoch_trainacc)+1),epoch_trainacc,lw=2,label="epoch_trainacc")
plt.plot(range(1,len(epoch_testacc)+1),epoch_testacc,lw=2,label="epoch_trainacc_limit")

plt.ylim((50,100))
plt.xlim((0,len(epoch_trainacc)))
plt.legend()
plt.show()

sheet = ExcelFile.sheet_by_name('epoch_testloss')
epoch_trainloss = sheet.col_values(0)
sheet = ExcelFile1.sheet_by_name('epoch_testloss')
epoch_testloss = sheet.col_values(0)


plt.xlabel("Training Epochs")
plt.ylabel("Training Loss")
plt.plot(range(1,len(epoch_trainloss)+1),epoch_trainloss,lw=2,label="epoch_testloss")
plt.plot(range(1,len(epoch_testloss)+1),epoch_testloss,lw=2,label="epoch_testloss_limit")

plt.ylim((0,0.9))
plt.xlim((0,len(epoch_trainloss)))
plt.legend()
plt.show()


sheet = ExcelFile.sheet_by_name('epoch_testacc')
epoch_trainacc = sheet.col_values(0)
sheet = ExcelFile1.sheet_by_name('epoch_testacc')
epoch_testacc = sheet.col_values(0)

epoch_trainacc = list(map(lambda x:x*100, epoch_trainacc))
epoch_testacc = list(map(lambda x:x*100, epoch_testacc))
plt.xlabel("Training Epochs")
plt.ylabel("Accuracy")
plt.plot(range(1,len(epoch_trainacc)+1),epoch_trainacc,lw=2,label="epoch_testacc")
plt.plot(range(1,len(epoch_testacc)+1),epoch_testacc,lw=2,label="epoch_testacc_limit")

plt.ylim((50,100))
plt.xlim((0,len(epoch_trainacc)))
plt.legend()
plt.show()