import pandas as pd
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
import os
import random
from sklearn.model_selection import train_test_split
from utils.util import write_to_txt

class DF:
    def __init__(self, args):
        self.normalization = True
        self.normalization_method = args.normalization_method  # min-max, z-score
        self.args = args

    def _3_sigma(self, Ser1):
        '''
        Identify outliers using 3-sigma rule.
        :param Ser1: pandas Series
        :return: indices of outliers
        '''
        rule = (Ser1.mean() - 3 * Ser1.std() > Ser1) | (Ser1.mean() + 3 * Ser1.std() < Ser1)
        index = np.arange(Ser1.shape[0])[rule]
        return index

    def delete_3_sigma(self, df):
        '''
        Remove outliers from DataFrame.
        :param df: pandas DataFrame
        :return: DataFrame without outliers
        '''
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()
        df = df.reset_index(drop=True)
        out_index = []
        for col in df.columns:
            index = self._3_sigma(df[col])
            out_index.extend(index)
        out_index = list(set(out_index))
        df = df.drop(out_index, axis=0)
        df = df.reset_index(drop=True)
        return df

    def read_one_csv(self, file_name, nominal_capacity=None):
        '''
        Read a CSV file and return a processed DataFrame.
        :param file_name: str, path to CSV
        :param nominal_capacity: float, for SOH calculation
        :return: pandas DataFrame
        '''
        df = pd.read_csv(file_name)
        df.insert(df.shape[1] - 1, 'cycle index', np.arange(df.shape[0]))
        df = self.delete_3_sigma(df)

        if nominal_capacity is not None:
            df['capacity'] = df['capacity'] / nominal_capacity
            f_df = df.iloc[:, :-1]
            if self.normalization_method == 'min-max':
                f_df = 2 * (f_df - f_df.min()) / (f_df.max() - f_df.min()) - 1
            elif self.normalization_method == 'z-score':
                f_df = (f_df - f_df.mean()) / f_df.std()
            df.iloc[:, :-1] = f_df
        return df

    def load_one_battery(self, path, nominal_capacity=None):
        '''
        Read a CSV and split into input/output pairs.
        :param path: str, path to CSV
        :param nominal_capacity: float
        :return: tuples (x1, y1), (x2, y2)
        '''
        df = self.read_one_csv(path, nominal_capacity)
        x = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values
        x1 = x[:-1]
        x2 = x[1:]
        y1 = y[:-1]
        y2 = y[1:]
        return (x1, y1), (x2, y2)

    def load_all_battery(self, path_list, nominal_capacity):
        '''
        Read multiple CSVs and create DataLoaders.
        :param path_list: list of file paths
        :param nominal_capacity: float
        :return: dict of DataLoaders (train, valid, test, train_2, valid_2, test_3)
        '''
        X1, X2, Y1, Y2 = [], [], [], []
        if self.args.log_dir is not None and self.args.save_folder is not None:
            save_name = os.path.join(self.args.save_folder, self.args.log_dir)
            write_to_txt(save_name, 'data path:')
            write_to_txt(save_name, str(path_list))
        for path in path_list:
            (x1, y1), (x2, y2) = self.load_one_battery(path, nominal_capacity)
            X1.append(x1)
            X2.append(x2)
            Y1.append(y1)
            Y2.append(y2)

        X1 = np.concatenate(X1, axis=0)
        X2 = np.concatenate(X2, axis=0)
        Y1 = np.concatenate(Y1, axis=0)
        Y2 = np.concatenate(Y2, axis=0)

        tensor_X1 = torch.from_numpy(X1).float()
        tensor_X2 = torch.from_numpy(X2).float()
        tensor_Y1 = torch.from_numpy(Y1).float().view(-1, 1)
        tensor_Y2 = torch.from_numpy(Y2).float().view(-1, 1)

        # Condition 1: Split into train (80%), test (20%), then train into train (80%), valid (20%)
        split = int(tensor_X1.shape[0] * 0.8)
        train_X1, test_X1 = tensor_X1[:split], tensor_X1[split:]
        train_X2, test_X2 = tensor_X2[:split], tensor_X2[split:]
        train_Y1, test_Y1 = tensor_Y1[:split], tensor_Y1[split:]
        train_Y2, test_Y2 = tensor_Y2[:split], tensor_Y2[split:]
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(train_X1, train_X2, train_Y1, train_Y2, test_size=0.2, random_state=420)

        train_loader = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                  batch_size=self.args.batch_size, shuffle=True)
        valid_loader = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                  batch_size=self.args.batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(test_X1, test_X2, test_Y1, test_Y2),
                                 batch_size=self.args.batch_size, shuffle=False)

        # Condition 2: Split directly into train (80%), valid (20%)
        train_X1, valid_X1, train_X2, valid_X2, train_Y1, valid_Y1, train_Y2, valid_Y2 = \
            train_test_split(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2, test_size=0.2, random_state=420)
        train_loader_2 = DataLoader(TensorDataset(train_X1, train_X2, train_Y1, train_Y2),
                                    batch_size=self.args.batch_size, shuffle=True)
        valid_loader_2 = DataLoader(TensorDataset(valid_X1, valid_X2, valid_Y1, valid_Y2),
                                    batch_size=self.args.batch_size, shuffle=True)

        # Condition 3: No split, full dataset as test
        test_loader_3 = DataLoader(TensorDataset(tensor_X1, tensor_X2, tensor_Y1, tensor_Y2),
                                   batch_size=self.args.batch_size, shuffle=False)

        loader = {'train': train_loader, 'valid': valid_loader, 'test': test_loader,
                  'train_2': train_loader_2, 'valid_2': valid_loader_2, 'test_3': test_loader_3}
        return loader

class MITdata(DF):
    def __init__(self, root='data/MIT_data', args=None):
        super(MITdata, self).__init__(args)
        self.root = root
        self.batchs = ['2017-05-12', '2017-06-30', '2018-04-12']
        if self.normalization:
            self.nominal_capacity = 1.1
        else:
            self.nominal_capacity = None

    def read_one_batch(self, batch):
        '''
        Read a batch of CSV files.
        :param batch: int, one of [1, 2, 3]
        :return: dict of DataLoaders
        '''
        assert batch in [1, 2, 3], 'batch must be in [1, 2, 3]'
        root = os.path.join(self.root, self.batchs[batch - 1])
        file_list = os.listdir(root)
        path_list = [os.path.join(root, file) for file in file_list]
        return self.load_all_battery(path_list=path_list, nominal_capacity=self.nominal_capacity)

    def read_all(self, specific_path_list=None):
        '''
        Read all or specified CSV files and create DataLoaders.
        :param specific_path_list: list of file paths, or None to read all
        :return: dict of DataLoaders
        '''
        if specific_path_list is None:
            file_list = []
            for batch in self.batchs:
                root = os.path.join(self.root, batch)
                files = os.listdir(root)
                for file in files:
                    path = os.path.join(root, file)
                    file_list.append(path)
            return self.load_all_battery(path_list=file_list, nominal_capacity=self.nominal_capacity)
        else:
            return self.load_all_battery(path_list=specific_path_list, nominal_capacity=self.nominal_capacity)

if __name__ == '__main__':
    import argparse

    def get_args():
        parser = argparse.ArgumentParser()
        parser.add_argument('--data', type=str, default='MIT', help='MIT')
        parser.add_argument('--batch', type=int, default=1, help='1, 2, 3')
        parser.add_argument('--batch_size', type=int, default=256, help='batch size')
        parser.add_argument('--normalization_method', type=str, default='z-score', help='min-max, z-score')
        parser.add_argument('--log_dir', type=str, default='test.txt', help='log dir')
        return parser.parse_args()

    args = get_args()
    mit = MITdata(root='data/MIT_data', args=args)
    mit.read_one_batch(batch=1)
    loader = mit.read_all()
    train_loader = loader['train']
    test_loader = loader['test']
    valid_loader = loader['valid']
    all_loader = loader['test_3']
    print('train_loader:', len(train_loader), 'test_loader:', len(test_loader),
          'valid_loader:', len(valid_loader), 'all_loader:', len(all_loader))

    for iter, (x1, x2, y1, y2) in enumerate(train_loader):
        print('x1 shape:', x1.shape)
        print('x2 shape:', x2.shape)
        print('y1 shape:', y1.shape)
        print('y2 shape:', y2.shape)
        print('y1 max:', y1.max())
        break