import os
path = "/DATA/LWN/new experiment2（复件）/data/SARS-Cov-TEST/test/"
# path = "C:/Users/wurenzhong/Desktop/data"
filelist = []
for root, dirs, files in os.walk(path):
    print(dirs)
    for file in files:
        file = os.path.join(root, file)
        print(file)
        with open('./SARS-Cov-test.txt', 'a') as fileobject:
            fileobject.writelines(file+'\n')


# filename='data_batch_1.mat';
# data=[];
# labels=[];
# for i=1:5
#     file=matfile(filename);
#     data=[data;file.data];
#     labels=[labels;file.labels];
#     filename(12)=int2str(i+1);
# end
# save('train.mat','data','labels')
