import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
# from sklearn.preprocessing import StandardScaler

from utils.tools import StandardScaler
from utils.timefeatures import time_features

import warnings
warnings.filterwarnings('ignore')

class Dataset_ETT_ms(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,
                 train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, debug=False):
        # size [seq_len, label_len, pred_len]
        # info
        #import ipdb; ipdb.set_trace()
        self.debug = debug
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0] # 1152
            self.label_len = size[1] # 1152 
            self.pred_len = size[2] # 576, e.g., 2 slices to predict 1 slice
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path # './data/ETT/'
        self.data_path = data_path # 'archxixia_ms.csv'

        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

        self.__read_data__()
        #import ipdb; ipdb.set_trace()

    def __read_data__(self):
        #import ipdb; ipdb.set_trace()
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, # (124416, 3), row=一共多少个“ms points”，column=3，"date, slice.num, value"
                                          self.data_path))

        # TODO need to change this!
        total_line_num = df_raw.shape[0]
        point1 = int(total_line_num * self.train_ratio)
        point2 = int(total_line_num * (self.train_ratio + self.test_ratio))
        point3 = total_line_num

        #border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len] # [0, 8304, 11184]
        #border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24] # [8640, 11520, 14400]
        
        border1s = [0, point1 - self.seq_len, point2 - self.seq_len] # [0, 98380, 110822]
        border2s = [point1, point2, point3] # [99532, 111974, 124416]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS' or self.features=='ms':
            cols_data = df_raw.columns[2:] # Index(['OT'], dtype='object') 
            df_data = df_raw[cols_data] #  (124416, 1)
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # df_data[0:99532], 
            self.scaler.fit(train_data.values) # [99532, 1], train data
            data = self.scaler.transform(df_data.values) # TODO changed! (x - mean.of.train.data)/std.of.train.data
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2] # train=(99532, 1); dev=(13594, 1); test=(13594, 1). 关于date的信息，精确到了yyyy-mm-dd hh:mm:ss.575000
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq) # train=[99532, 7]; dev=(13594, 7); test=(13594, 7), 用7个特征值来表示一个时间点(to ms)

        self.data_x = data[border1:border2] # data: [124416, 1], border1=0, border2=99532, so self.data_x=[99532, 1] for train. | (13594,1) for dev | (13594, 1) for test  
        if self.inverse: # False
            self.data_y = df_data.values[border1:border2] 
        else:
            self.data_y = data[border1:border2] # train=(99532, 1); dev=(13594, 1); test=(13594,1)
        self.data_stamp = data_stamp # train=(99532, 7); dev=(13594,7); test=(13594, 7)
        #import ipdb; ipdb.set_trace()
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len # self.seq_len=336=24*14
        r_begin = s_end - self.label_len # self.label_len=336=24*14 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #import ipdb; ipdb.set_trace()
        return seq_x, seq_y, seq_x_mark, seq_y_mark # seq_x:[336,7], seq_y:[336+168,7] and seq_x=seq_y[:336]; seq_x_mark:[336,4], seq_y_mark:[504,4]
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1 # 8640 - 336 - 168 + 1 = 8137

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def inverse_transform_tar(self, data):
        return self.scaler.inverse_transform_tar(data)

class Dataset_ETT_hour2(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None,
                 train_ratio=0.8, dev_ratio=0.1, test_ratio=0.1, debug=False):
        # size [seq_len, label_len, pred_len]
        # info
        self.debug = debug
        if self.debug:
            import ipdb; ipdb.set_trace()
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0] # 1152
            self.label_len = size[1] # 1152 
            self.pred_len = size[2] # 576, e.g., 2 slices to predict 1 slice
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path # './data/ETT/'
        self.data_path = data_path # 'archxixia_ms.csv'

        self.train_ratio = train_ratio
        self.dev_ratio = dev_ratio
        self.test_ratio = test_ratio

        self.__read_data__()
        if self.debug:
            import ipdb; ipdb.set_trace()

    def __read_data__(self):
        if self.debug:
            import ipdb; ipdb.set_trace()
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, # (124416, 3), row=一共多少个“ms points”，column=3，"date, slice.num, value"
                                          self.data_path))

        # TODO need to change this!
        total_line_num = df_raw.shape[0]
        point1 = int(total_line_num * self.train_ratio)
        point2 = int(total_line_num * (self.train_ratio + self.test_ratio))
        point3 = total_line_num

        #border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len] # [0, 8304, 11184]
        #border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24] # [8640, 11520, 14400]
        
        border1s = [0, point1 - self.seq_len, point2 - self.seq_len] # [0, 98380, 110822]
        border2s = [point1, point2, point3] # [99532, 111974, 124416]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS' or self.features=='ms':
            cols_data = df_raw.columns[1:] # Index(['OT'], dtype='object') 
            df_data = df_raw[cols_data] #  (124416, 1)
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # df_data[0:99532], 
            self.scaler.fit(train_data.values) # [99532, 1], train data
            data = self.scaler.transform(df_data.values) # TODO changed! (x - mean.of.train.data)/std.of.train.data
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2] # train=(99532, 1); dev=(13594, 1); test=(13594, 1). 关于date的信息，精确到了yyyy-mm-dd hh:mm:ss.575000
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq) # train=[99532, 7]; dev=(13594, 7); test=(13594, 7), 用7个特征值来表示一个时间点(to ms)

        self.data_x = data[border1:border2] # data: [124416, 1], border1=0, border2=99532, so self.data_x=[99532, 1] for train. | (13594,1) for dev | (13594, 1) for test  
        if self.inverse: # False
            self.data_y = df_data.values[border1:border2] 
        else:
            self.data_y = data[border1:border2] # train=(99532, 1); dev=(13594, 1); test=(13594,1)
        self.data_stamp = data_stamp # train=(99532, 7); dev=(13594,7); test=(13594, 7)
        if self.debug:
            import ipdb; ipdb.set_trace()
    
    def __getitem__(self, index):
        if self.debug:
            import ipdb; ipdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len # self.seq_len=336=24*14
        r_begin = s_end - self.label_len # self.label_len=336=24*14 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        if self.debug:
            import ipdb; ipdb.set_trace()
        return seq_x, seq_y, seq_x_mark, seq_y_mark # seq_x:[336,7], seq_y:[336+168,7] and seq_x=seq_y[:336]; seq_x_mark:[336,4], seq_y_mark:[504,4]
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1 # 8640 - 336 - 168 + 1 = 8137

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
    
    def inverse_transform_tar(self, data):
        return self.scaler.inverse_transform_tar(data)

class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        #import ipdb; ipdb.set_trace()
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0] # 336 = 24*14, 14 days
            self.label_len = size[1] # 336
            self.pred_len = size[2] # 168 = 24*7, 7 days
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path # './data/ETT/'
        self.data_path = data_path # 'ETTh2.csv'
        self.__read_data__()
        #import ipdb; ipdb.set_trace()

    def __read_data__(self):
        #import ipdb; ipdb.set_trace()
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, # (17420, 8), row=一共多少个“小时点”，column=8，date以及7个features
                                          self.data_path))

        # TODO need to change this!
        border1s = [0, 12*30*24 - self.seq_len, 12*30*24+4*30*24 - self.seq_len] # [0, 8304, 11184]
        border2s = [12*30*24, 12*30*24+4*30*24, 12*30*24+8*30*24] # [8640, 11520, 14400]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS' or self.features=='ms':
            cols_data = df_raw.columns[1:] # (7,), Index(['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT'], dtype='object')
            df_data = df_raw[cols_data] # (17420, 7) 
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]] # df_data[0:8640], 8640=12*30*24 是12个月，每个月30天，每天24小时得到的
            self.scaler.fit(train_data.values) # [8640, 7], 一年的数据作为训练data
            data = self.scaler.transform(df_data.values) # TODO changed! from 41.130001 to -0.03893579, but how?
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2] # (8640, 1) 关于date的信息，精确到了yyyy-mm-dd hh:mm:ss
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq) # [8640, 4], 用四个特征值来表示一个时间点, hour-of-day, day-of-week/month/year

        self.data_x = data[border1:border2] # data: [17420, 7], border1=0, border2=8640, so self.data_x=[8640, 7], [[-0.03893579,  0.04524569, -0.59755963] 
        if self.inverse: # False
            self.data_y = df_data.values[border1:border2] # self.data_y=[8640, 7], [-0.03893579,  0.04524569, -0.59755963]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        #import ipdb; ipdb.set_trace()
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len # self.seq_len=336=24*14
        r_begin = s_end - self.label_len # self.label_len=336=24*14 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #import ipdb; ipdb.set_trace()
        return seq_x, seq_y, seq_x_mark, seq_y_mark # seq_x:[336,7], seq_y:[336+168,7] and seq_x=seq_y[:336]; seq_x_mark:[336,4], seq_y_mark:[504,4]
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1 # 8640 - 336 - 168 + 1 = 8137

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_ETT_minute(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTm1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='t', cols=None):
        #import ipdb; ipdb.set_trace()
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        #import ipdb; ipdb.set_trace()

    def __read_data__(self):
        #import ipdb; ipdb.set_trace()
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12*30*24*4 - self.seq_len, 12*30*24*4+4*30*24*4 - self.seq_len]
        border2s = [12*30*24*4, 12*30*24*4+4*30*24*4, 12*30*24*4+8*30*24*4]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        
        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        #import ipdb; ipdb.set_trace()
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #import ipdb; ipdb.set_trace()
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='h', cols=None):
        #import ipdb; ipdb.set_trace()
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['train', 'test', 'val']
        type_map = {'train':0, 'val':1, 'test':2}
        self.set_type = type_map[flag]
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        #import ipdb; ipdb.set_trace()

    def __read_data__(self):
        #import ipdb; ipdb.set_trace()
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        # cols = list(df_raw.columns); 
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]

        num_train = int(len(df_raw)*0.7)
        num_test = int(len(df_raw)*0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [0, num_train-self.seq_len, len(df_raw)-num_test-self.seq_len]
        border2s = [num_train, num_train+num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        #import ipdb; ipdb.set_trace()
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len 
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate([self.data_x[r_begin:r_begin+self.label_len], self.data_y[r_begin+self.label_len:r_end]], 0)
        else:
            seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #import ipdb; ipdb.set_trace()
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len- self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

class Dataset_Pred(Dataset):
    def __init__(self, root_path, flag='pred', size=None, 
                 features='S', data_path='ETTh1.csv', 
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        # size [seq_len, label_len, pred_len]
        # info
        #import ipdb; ipdb.set_trace()
        if size == None:
            self.seq_len = 24*4*4
            self.label_len = 24*4
            self.pred_len = 24*4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']
        
        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols=cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()
        #import ipdb; ipdb.set_trace()

    def __read_data__(self):
        #import ipdb; ipdb.set_trace()
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols=self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns); cols.remove(self.target); cols.remove('date')
        df_raw = df_raw[['date']+cols+[self.target]]
        
        border1 = len(df_raw)-self.seq_len
        border2 = len(df_raw)
        
        if self.features=='M' or self.features=='MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features=='S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values
            
        tmp_stamp = df_raw[['date']][border1:border2]
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len+1, freq=self.freq)
        
        df_stamp = pd.DataFrame(columns = ['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq[-1:])

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp
        #import ipdb; ipdb.set_trace()
    
    def __getitem__(self, index):
        #import ipdb; ipdb.set_trace()
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin+self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin+self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        #import ipdb; ipdb.set_trace()
        return seq_x, seq_y, seq_x_mark, seq_y_mark
    
    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
