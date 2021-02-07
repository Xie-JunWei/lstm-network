# make label 打标签
import shutil
import csv
import os  #导入os模块
def create_csv(path):
    with open(path, "w+", newline='') as file:
        csv_file = csv.writer(file)
        head = ["file_name","label"]
        csv_file.writerow(head)
def append_csv(path,File_name,Label):
    with open(path, "a+", newline='') as file: # 处理csv读写时不同换行符  linux:\n    windows:\r\n    mac:\r
        csv_file = csv.writer(file)
        datas = [[File_name, Label]]
        csv_file.writerows(datas)

def read_csv(path):
    with open(path,"r+") as file:
        csv_file = csv.reader(file)
        for data in csv_file:
            print("data:", data)

rawdata='F:\HXDD/'
save_train_path = r'F:\classi\train/'
save_test_path = r'F:\classi\test/'
train_csv_name = 'dataset_train.csv'
test_csv_name = 'dataset_test.csv'
create_csv(os.path.join(rawdata, train_csv_name))
create_csv(os.path.join(rawdata, test_csv_name))
folder_name={'LvsD','const_aphi','const_h'}
dic = {'LvsD': 0, 'const_aphi': 1, 'const_h': 2}
# label_file='test'
for label_file in folder_name:
    use_path=os.path.join(rawdata, label_file)
    for f in os.listdir(use_path):    #获取所有image目录下的文件
        print(f)
        #进行改名
        file_name=str(label_file)+ str(f)
        count=int(str(f).split('.')[0])

        if count<9000:
            shutil.copyfile(os.path.join(use_path, f), os.path.join(save_train_path, file_name))
            append_csv(os.path.join(rawdata, train_csv_name),file_name,str(dic[label_file]))
        else:
            shutil.copyfile(os.path.join(use_path, f), os.path.join(save_test_path, file_name))
            append_csv(os.path.join(rawdata, test_csv_name), file_name, str(dic[label_file]))

