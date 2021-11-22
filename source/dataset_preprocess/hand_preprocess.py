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
        os.system('rm -r ' + config['graph_test_dataset'])
        os.makedirs(config['graph_test_dataset'])
        self.read_and_write(config['raw_test_dataset'], config['graph_test_dataset'], config['max_len'], config['segment'])
        print("Done!")

        print("Preprocessing train dataset...")
        os.system('rm -r ' + config['graph_train_dataset'])
        os.makedirs(config['graph_train_dataset'])
        self.read_and_write(config['raw_train_dataset'], config['graph_train_dataset'], config['max_len'], config['segment'])
        print("Done!")

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
                    new_line, threshold = self.preprocess(line)
                    if len(new_line) < config["length_threshold"]:
                        continue
                    tmp = self.segment(new_line, max_length, segment)
                    if file in self.abnormal_files:
                        label = '.1'
                    else:
                        label = '.0'
                    # pdb.set_trace()
                    name = file.split('-', 1)[0]
                    date = file.split('-', 1)[1]
                    if name in self.abnormal_users:
                        label = '.1' + label
                    else:
                        label = '.0' + label
                    userindex = self.username2index[name]
                    dateindex = self.date2index[date]
                    for i in range(len(tmp)):
                        with open (output_file_path + "/" + str(userindex) + '.' + str(
                                dateindex) + '.' + str(i) + label + '.json','w') as f:
                            json.dump([threshold, tmp[i][0], tmp[i][1]], f)

    def left_sep_right(self, line):
        tmp_line = copy.deepcopy(line)
        left = []
        right = []
        for x in tmp_line:
            if x[-1]==2:
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

        max_x = line[0][1][0][0]
        max_y = line[0][1][0][1]
        min_x = line[0][1][0][0]
        min_y = line[0][1][0][1]
        start_line = line[0][1][0][2]
        for event in line:
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
        for event in line:
            if event[1][0][2]<cur_time[0]:
                hand_label = 1
            elif event[1][0][2]<cur_time[1]:
                hand_label = 0
            else:
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

            for point in event[1]:
                if point[0]<0 or point[1]<0:
                    continue
                new_line.append([point[0], point[1], point[2]-start_line, event[0]+1, hand_label+1])
                cur_time[hand_label] = point[2]
        new_line[0][2] = 0
        return new_line, [max_x, max_y, min_x, min_y]
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

if __name__ == '__main__':

    config_path = "../config.json"
    check_path = "../public_filter/check.json"
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