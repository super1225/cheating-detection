import os

for i in range(6, 7):
    print("python pretrain_and_train.py --train_mode "+str(i) + " --load_file train.initial --epochs 20")
    for j in range(0, 30):
        print('python bert_sne.py --train_mode 0 --load_file bert.task'+str(i)+'.ep'+str(j))
        # print('python bert_sne.py --train_mode 3 --load_file bert.task'+str(i)+'.ep'+str(j))
        print('python bert_sne.py --train_mode 2 --load_file bert.task'+str(i)+'.ep'+str(j))