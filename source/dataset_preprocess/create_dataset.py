import json
import os
import pdb
from random import random

config_path = "../config.json"
with open(config_path) as f:
    config = json.load(f)

def username_to_index(file_directory_1, file_directory_2, output_path):
    username_to_index = dict()
    username_counter = dict()
    date_to_index = dict()
    files1 = os.listdir(file_directory_1)
    files2 = os.listdir(file_directory_2)
    files = files1 + files2
    for file in files:
        name = file.split('-', 1)[0]
        # pdb.set_trace()
        date = file.split('-', 1)[1]

        if name not in username_to_index.keys():
            new_index = len(username_to_index)
            username_to_index[name] = new_index
            username_counter[new_index] = 1
        else:
            username_counter[username_to_index[name]] += 1

        if date not in date_to_index.keys():
            new_index = len(date_to_index)
            date_to_index[date] = new_index

    with open(output_path + '/username2index.json', 'w') as f:
        json.dump(username_to_index, f)

    with open(output_path + '/username_counter.json', 'w') as f:
        json.dump(username_counter, f)

    with open(output_path + '/date2index.json', 'w') as f:
        json.dump(date_to_index, f)

def sample_dataset():
    # username_to_index(config['raw_test_dataset'], config['raw_train_dataset'], config['identity_path'])
    # with open(config['identity_path'] + '/username2index.json','r') as f:
    #     username2index = json.load(f)

    # with open(config['identity_path'] + '/username_counter.json', 'r') as f:
    #     username_counter = json.load(f)

    # with open(config['identity_path'] + '/date2index.json', 'r') as f:
    #     date2index = json.load(f)

    files2 = os.listdir(config['raw_test_dataset'])
    print(len(files2))

    with open(config['identity_path'] + '/more_than_one.json', 'r') as f:
        more_than_one = json.load(f)
    counter_test = 0
    counter_train = 0
    train = []
    test = []
    for x in more_than_one:
        p = random()
        q = random()
        if p < 0.05 and q < 0.8 and counter_train < 4000:
            train.append(x)
            counter_train += 1
        if p < 0.05 and q > 0.8 and counter_test < 1000:
            test.append(x)
            counter_test += 1

    print('creating test dataset...')
    for x in test:
        os.system('cp ' + config['raw_test_dataset'] + '/' + x + '* ' + config['identity_test_path'])
        os.system('cp ' + config['raw_train_dataset'] + '/' + x + '* ' + config['identity_test_path'])
    print('completed!')

    print('creating train dataset...')
    for x in train:
        os.system('cp ' + config['raw_test_dataset'] + '/' + x + '* ' + config['identity_train_path'])
        os.system('cp ' + config['raw_train_dataset'] + '/' + x + '* ' + config['identity_train_path'])
    print('completed!')


def copy_with_threshold_of_hands(src, dst):
    max_sep = 0
    min_sep = 0
    sep = 0
    avg = 0
    files2 = os.listdir(src)
    print(len(files2))
    for file in files2:
        if_copy = True
        with open(src + '/' + file, 'r') as f:
            line = json.load(f)
            if sep % 100 == 0:
                print(sep)

            if len(line[1]) - line[2] > 130:
                if_copy = False
                min_sep += 1
            elif line[2] > 270:
                if_copy = False
                max_sep += 1
            else:
                sep += 1
            avg += line[2]
        f.close()
        if if_copy:
            os.system('cp ' + src + '/' + file + ' ' + dst)

    print(max_sep, min_sep, sep, avg / sep)


# copy_with_threshold_of_hands(config['hand_train_dataset'], config['graph_train_dataset'])
# copy_with_threshold_of_hands(config['hand_test_dataset'], config['graph_test_dataset'])

copy_with_threshold_of_hands(config['plot_dataset'], config['graph_plot_dataset'])

