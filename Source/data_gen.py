import csv
import torch
import numpy as np
from sklearn.model_selection import train_test_split

class Data():

    def __init__(self, path_to_data: str, test_size: int, verbose=True):
        '''Data generation
        x_* attributes contain n_bits of grayscale pixels representing some digit
        y_* attributes are the corresponding digits to x_*
        '''

        with open(path_to_data,'r') as d:
            data_iter = csv.reader(d, delimiter =' ')

            data = []
            for e in data_iter:
            	e.pop(8)
            	data.append(list(map(int, e)))

            data_array = np.array(data, dtype=np.int)


        # Split data into i/o and normalize
        x_ = data_array[:, :8]
        y_ = data_array[:, 8:]

        x_train, x_test, y_train, y_test = train_test_split(
            x_, y_, test_size=test_size, shuffle=True)


        if verbose:            
            print('x_train.shape:', x_train.shape)   # (200, 8)
            print('y_train.shape:', y_train.shape)   # (200, 5)
            print('x_test.shape:', x_test.shape)     # (56, 8)
            print('y_test.shape:', y_test.shape)     # (56, 5)

            #print(x_train)
            #print(y_train)
            print(x_test)
            print(y_test)

        self.x_train = torch.tensor(x_train, dtype=torch.float)
        self.x_test  = torch.tensor(x_test, dtype=torch.float)
        self.y_train = torch.tensor(y_train, dtype=torch.long)
        self.y_test  = torch.tensor(y_test, dtype=torch.long)

if __name__ == '__main__':

    print('Hello, world!')
    Data('../Datasets/d=3.txt', 56)
