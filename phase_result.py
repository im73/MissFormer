import os
import numpy as np
root_path = '/home/LAB/zhuqs/prj/gmlp_split/log'
files = os.listdir(root_path)

for file in files:
    full_path = os.path.join(root_path, file)
    f = open(full_path, 'r+')
    texts = f.readlines()
    print("filename : {},".format(file), end = '')
    mse_list = []
    mae_list = []
    for line in texts:
        if line[:3] == 'mse':
            key_word = line.split(', ')
            mse = float(key_word[0].split(':')[-1])
            mae = float(key_word[1].split(':')[-1])

            mse_list.append(mse)
            mae_list.append(mae)
    per_exp = 5
    for i in range(len(mse_list) // per_exp):
        print("{},{},".format(np.mean(np.array(mse_list[i*per_exp:(i+1)*per_exp])), np.mean(np.array(mae_list[i*per_exp:(i+1)*per_exp]))), end = '')
    print('\n')

