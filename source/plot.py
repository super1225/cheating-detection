import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import numpy as np

def plot_loss():
    y = []
    data = []
    with open("/home/syy/project/traj_supervised/output/result_0_test.txt", "r") as f:  
        while 1:
            line = f.readline()
            if not line:
                break
            data.append(float(line[line.rfind('loss=')+5:line.rfind('loss=')+11]))
    # enc = torch.load('D:\MobileNet_v1\plan1-AddsingleLayer\loss\epoch_{}'.format(i))
    tempy = data
    #print(tempy)
    y += tempy
    x = range(0,len(y))
    #mark_on = [0,50]
    plt.plot(x, y, '.-', markevery=100)
    plt_title = 'loss'
    plt.title(plt_title)
    plt.xlabel('epoch')
    plt.ylabel('LOSS')
    plt.savefig('/home/syy/project/traj_supervised/output/loss.jpg')
    plt.show()

if __name__ == "__main__":
    plot_loss()