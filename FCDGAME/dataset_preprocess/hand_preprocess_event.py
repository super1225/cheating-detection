import os
import torch
import json
import csv
import pdb
import copy

class Preprocess:
    """
    Read trajectory files and preprocess data, and following 
    information are provided:
        1. grid information of the current points
        2. time interval between the current point and the previous point
        3. event type
    """
    def __init__(self, config):
        """
        :param file_directory: directory path of the dataset to be loaded
        :param max_length: maximum number of points in one trajectory
        :param segment: 
            1.if True, then truncate the trajectory longer than max_length into segments
            2.otherwise False, then just drop points exceeding the max_length
        """
        csv_reader = csv.reader(open(config['label_filepath']))
        self.abnormal_files = [x[0]+'-'+x[1]+'.json' for x in csv_reader]
        csv_reader = csv.reader(open(config['label_filepath']))
        self.abnormal_users = [x[0] for x in csv_reader]#带日期
        self.config = config
        with open(config['identity_path']+'/username2index.json', 'r') as f:
            self.username2index = json.load(f)

        with open(config['identity_path']+'/date2index.json', 'r') as f:
            self.date2index = json.load(f)

        print("Preprocessing test dataset...")
        os.system('rm -r ' + config['test_dataset'])
        os.makedirs(config['test_dataset'])
        self.read_and_write(config['raw_test_dataset'], config['test_dataset'], config['max_len'], config['segment'])
        print("Done!")

        # print("Preprocessing train dataset...")
        # os.system('rm -r ' + config['train_dataset'])
        # os.makedirs(config['train_dataset'])
        # self.read_and_write(config['raw_train_dataset'], config['train_dataset'], config['max_len'], config['segment'])
        # print("Done!")

    def read_and_write(self, file_directory, output_file_path, max_length, segment):
        """
        read trajectory files and segment each trajectory or drop points
        """
        tmp = []
        files= os.listdir(file_directory)
        for file in files:
            if not os.path.isdir(file):
                with open(file_directory + "/" + file) as f:
                    line = json.load(f)
                    tmp = []
                    new_line, threshold = self.preprocess(line)#每个坐标形式：[x,y,时间戳（减去该人的轨迹的第一个坐标点的时间戳），事件标签+1，事件轨迹hand_lable(同一个事件轨迹的左右手lable一样)】
                    if len(new_line) < config["length_threshold"]:
                        continue
                    tmp = self.segment(new_line, max_length, segment)#返回分段之后的轨迹，每条轨迹的形式为（左手轨迹+右手轨迹，左手轨迹长度）
                    if file in self.abnormal_files:#文件标签为异常
                        label = '.1'
                    else:
                        label = '.0'
                    # pdb.set_trace()
                    name = file.split('-', 1)[0]#获取当前文件的用户名
                    date = file.split('-', 1)[1]#过去当前文件的日期
                    if name in self.abnormal_users:#存在过异常的用户
                        label = '.1' + label
                    else:
                        label = '.0' + label
                    userindex = self.username2index[name]#获取用户名对应的编号
                    dateindex = self.date2index[date]#获取日期对应的编号
                    for i in range(len(tmp)):
                        with open (output_file_path + "/" + str(userindex) + '.' + str(
                                dateindex) + '.' + str(i) + label + '.json','w') as f:
                            json.dump([threshold, tmp[i][0], tmp[i][1]], f)

    def left_sep_right(self, line):#将轨迹变成左手+右手的形式
        tmp_line = copy.deepcopy(line)
        left = []
        right = []
        for x in tmp_line:
            if x[-1]==2:#左右手标签
                left.append(x)
            else:
                right.append(x)
        if len(left)>self.config['left_threshold'] or len(right)>self.config['right_threshold']:
            return None
        else:
            return [left+right, len(left)]

    def segment(self, line, max_length, segment):
        """
        truncate the trajectory into several segmentation according to maximum length
        """
        if segment:
            m = int(len(line)/max_length)
            tmp = []
            for i in range(0, m):
                lsr = self.left_sep_right(line[i*max_length:(i+1)*max_length])
                if lsr!=None:
                    tmp.append(lsr)
            if len(line) > max_length*m:
                lsr = self.left_sep_right(line[-max_length:])
                if lsr!=None:
                    tmp.append(lsr)
            return tmp
        else:
            if len(line)<max_length:
                lsr = self.left_sep_right(line)
                if lsr!=None:
                    return [lsr]
            else:
                lsr = self.left_sep_right(line[max_length])
                if lsr!=None:
                    return [lsr]

    def time_segmentation(self, t):
        """
        Segment the first order difference of timestamp
        """
        if t>self.config['time_segment']:
            return self.config['time_segment']
        elif t<0:
            return 0
        else:
            return int(t)

    def to_grids(self, max_x, max_y, point):
        """
        :param max_width: the maximum value of x-axis coordinate
        :param max_height: the maximum value of y-axis coordinate
        :param r_x: the width of each grid
        :param r_y: the height of each grid
        """
        return self.config['y_grid_nums']*int(self.config['x_grid_nums']*point[0]/max_x)+int(self.config['y_grid_nums']*point[1]/max_y)

    def preprocess(self, line):
        """
        Add a bool value to each point which represents left hand or right 
        hand, replace the timestamp of each point with first order difference 
        of timestamp of each hand's points.
        """

        max_x = line[0][1][0][0]#【轨迹】【事件轨迹】【事件标签，轨迹】【坐标】
        max_y = line[0][1][0][1]
        min_x = line[0][1][0][0]
        min_y = line[0][1][0][1]
        start_line = line[0][1][0][2]#第一个事件轨迹的第一个坐标的时间戳
        for event in line:#找到轨迹中x和y的做大和最小值
            for point in event[1]:
                if max_x < point[0]:
                    max_x = point[0]
                if min_x > point[0]:
                    min_x = point[0]
                    
                if max_y < point[1]:
                    max_y = point[1]
                if min_y > point[1]:
                    min_y = point[1]
        threshold = 0.6*min_x + 0.4*max_x

        cur_time = [0, 0]
        new_line = []
        
        '''
        变换前：[2, [[1675, 524, 1619506914197], [1824, 525, 1619506914597], [1795, 523, 1619506914964]]]
        变换后：[[1675, 524, 0, 3, 1], [1824, 525, 400, 3, 1], [1795, 523, 767, 3, 1]]
        '''
        for event in line:#一个事件轨迹对应一个左右手标签
            new_line_event = []
            if event[1][0][2]<cur_time[0]:#【事件轨迹】【坐标点】【时间戳】如果当前事件轨迹的时间戳小于hand_lable0的当前时间，那标签就是1
                hand_label = 1
            elif event[1][0][2]<cur_time[1]:
                hand_label = 0
            else:#统计事件轨迹中坐标分布在屏幕左右两边的数量
                hand = [0, 0]
                for point in event[1]:
                    if point[0]<threshold:
                        hand[0] += 1
                    else:
                        hand[1] += 1
                if hand[0]<hand[1]:
                    hand_label = 0
                else:
                    hand_label = 1

            for point in event[1]:#将该事件轨迹的形式变为【x,y,时间戳（减去该人的轨迹的第一个坐标点的时间戳），事件标签+1，事件轨迹hand_lable(同一个事件轨迹的左右手lable一样)】
                if point[0]<0 or point[1]<0:
                    continue
                new_line_event.append([point[0], point[1], point[2]-start_line, event[0]+1, hand_label+1])
                cur_time[hand_label] = point[2]
            new_line.append(new_line_event)
        new_line[0][0][2] = 0
        return new_line, [max_x, max_y, min_x, min_y]

if __name__ == '__main__':

    config_path = "../config.json"
    check_path = "../check.json"
    with open(config_path) as f:
        config = json.load(f)
    if False:
    # if os.path.exists(check_path):
        with open(check_path) as f:
            check = json.load(f)
        if check==config:
            print("Already preprocessed!")
        else:
            with open(check_path, 'w') as f:
                Preprocess(config)
                json.dump(config, f)
    else:
        Preprocess(config)
        with open(check_path, 'w') as f:
            json.dump(config, f)