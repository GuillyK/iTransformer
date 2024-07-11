import os
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle
from sklearn.utils.class_weight import compute_class_weight
from torch.nn.functional import pad
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, random_split
from tqdm import tqdm
from datetime import datetime

from utils.timefeatures import time_features


warnings.filterwarnings("ignore")


class Dataset_ETT_hour(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 - self.seq_len,
            12 * 30 * 24 + 4 * 30 * 24 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24,
            12 * 30 * 24 + 4 * 30 * 24,
            12 * 30 * 24 + 8 * 30 * 24,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_ETT_minute(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTm1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="t",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        border1s = [
            0,
            12 * 30 * 24 * 4 - self.seq_len,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - self.seq_len,
        ]
        border2s = [
            12 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 4 * 30 * 24 * 4,
            12 * 30 * 24 * 4 + 8 * 30 * 24 * 4,
        ]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Custom(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove("date")
        # groups = df_raw.groupby('FOI_ID_LEVERANCIER')

        df_raw = df_raw[["date"] + cols + [self.target]]

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[["date"]][border1:border2]
        df_stamp["date"] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_PEMS(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        data_file = os.path.join(self.root_path, self.data_path)
        data = np.load(data_file, allow_pickle=True)
        data = data["data"][:, :, 0]

        train_ratio = 0.6
        valid_ratio = 0.2
        train_data = data[: int(train_ratio * len(data))]
        valid_data = data[
            int(train_ratio * len(data)) : int(
                (train_ratio + valid_ratio) * len(data)
            )
        ]
        test_data = data[int((train_ratio + valid_ratio) * len(data)) :]
        total_data = [train_data, valid_data, test_data]
        data = total_data[self.set_type]

        if self.scale:
            self.scaler.fit(train_data)
            data = self.scaler.transform(data)

        df = pd.DataFrame(data)
        df = (
            df.fillna(method="ffill", limit=len(df))
            .fillna(method="bfill", limit=len(df))
            .values
        )

        self.data_x = df
        self.data_y = df

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Solar(Dataset):
    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        timeenc=0,
        freq="h",
    ):
        # size [seq_len, label_len, pred_len]
        # info
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        # init
        assert flag in ["train", "test", "val"]
        type_map = {"train": 0, "val": 1, "test": 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = []
        with open(
            os.path.join(self.root_path, self.data_path), "r", encoding="utf-8"
        ) as f:
            for line in f.readlines():
                line = line.strip("\n").split(",")
                data_line = np.stack([float(i) for i in line])
                df_raw.append(data_line)
        df_raw = np.stack(df_raw, 0)
        df_raw = pd.DataFrame(df_raw)

        num_train = int(len(df_raw) * 0.7)
        num_test = int(len(df_raw) * 0.2)
        num_valid = int(len(df_raw) * 0.1)
        border1s = [
            0,
            num_train - self.seq_len,
            len(df_raw) - num_test - self.seq_len,
        ]
        border2s = [num_train, num_train + num_valid, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        df_data = df_raw.values

        if self.scale:
            train_data = df_data[border1s[0] : border2s[0]]
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data)
        else:
            data = df_data

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = torch.zeros((seq_x.shape[0], 1))
        seq_y_mark = torch.zeros((seq_x.shape[0], 1))

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Pred(Dataset):
    def __init__(
        self,
        root_path,
        flag="pred",
        size=None,
        features="S",
        data_path="ETTh1.csv",
        target="OT",
        scale=True,
        inverse=False,
        timeenc=0,
        freq="15min",
        cols=None,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["pred"]

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        """
        df_raw.columns: ['date', ...(other features), target feature]
        """
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove("date")
        df_raw = df_raw[["date"] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == "M" or self.features == "MS":
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[["date"]][border1:border2]
        tmp_stamp["date"] = pd.to_datetime(tmp_stamp.date)
        pred_dates = pd.date_range(
            tmp_stamp.date.values[-1],
            periods=self.pred_len + 1,
            freq=self.freq,
        )

        df_stamp = pd.DataFrame(columns=["date"])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp["weekday"] = df_stamp.date.apply(
                lambda row: row.weekday(), 1
            )
            df_stamp["hour"] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp["minute"] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp["minute"] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(["date"], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(
                pd.to_datetime(df_stamp["date"].values), freq=self.freq
            )
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = self.data_x[r_begin : r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin : r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len + 1

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


class Dataset_Crop(Dataset):

    def __init__(
        self,
        root_path,
        flag="train",
        size=None,
        features="M",
        data_path="noordoostpolder_itransformer.csv",
        target="class_name_late",
        scale=True,
        timeenc=0,
        freq="d",
        var_seq_len=False,
        early_classification=False,
    ):
        # size [seq_len, label_len, pred_len]
        # info
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ["train", "val", "test"]
        self.flag = flag
        self.type_map = {"train": 0, "val": 1, "test": 2}
        self.split_id = self.type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.var_seq_len = var_seq_len
        self.early_classification = early_classification
        self.columns = None

        self.root_path = root_path
        self.data_path = data_path
        self.class_weights = None
        self.dates = 0
        self.__read_data__()

    def __read_data__(self):
        """
        This method reads the data from a CSV file and performs preprocessing steps such as scaling and feature engineering. It loads the data into the `data_x`, `data_y`, and `data_stamp` attributes of the class.

        Returns:
            None
        """
        self.scaler = StandardScaler()
        type_map = ["train", "val", "test"]
        type_sort = type_map[self.split_id]
        dataset_files = ["data_x.npy", "data_y.npy", "data_stamp.npy"]
        path = os.path.join(self.root_path, "Netherlands/")
        folder = os.path.join(path, type_sort)
        # check if in path train/test/val folder exists if not create
        if False:
            pass
        # if os.path.exists(folder):
        #     df_raw = pd.read_csv(os.path.join(path, self.data_path))
        #     target = df_raw.columns.str.startswith("class_name_late")
        #     self.target = list(df_raw.columns[target])
        #     self.class_weights = np.load(
        #         os.path.join(folder, "class_weights.npy")
        #     )
        #     self.data_x = np.load(os.path.join(folder, "data_x.npy"))
        #     self.data_y = np.load(os.path.join(folder, "data_y.npy"))
        #     self.data_stamp = np.load(os.path.join(folder, "data_stamp.npy"))
        #     self.masks = np.load(os.path.join(folder, "masks.npy"))
        #     self.seq_end_lengths = np.load(
        #         os.path.join(folder, "seq_end_lengths.npy")
        #     )
        #     self.dates = np.load(os.path.join(folder, "dates.npy"), allow_pickle=True)
        else:

            df_raw = pd.read_csv(os.path.join(path, self.data_path))

            """
                df_raw.columns: ['date', ...(other features), target feature]
                """
            cols = list(df_raw.columns)
            # target is columns that start with class_name_late (one hot encoding)
            target = df_raw.columns.str.startswith("class_name_late")
            print("TARGET", target)
            print(list(df_raw.columns[target]))
            self.target = list(df_raw.columns[target])
            # cols.remove(self.target)
            cols = [x for x in cols if x not in self.target]
            cols.remove("date")
            cols.remove("FOI_ID_LEVERANCIER")
            if self.scale:
                self.scaler.fit(df_raw[cols].values)
                scaled_data = self.scaler.transform(df_raw[cols].values)
            else:
                scaled_data = df_raw[cols].values
            df_raw[cols] = scaled_data
            # groups = df_raw.groupby("FOI_ID_LEVERANCIER")
            # print("these are the groups\n", groups)
            df_raw = df_raw[
                ["date"] + ["FOI_ID_LEVERANCIER"] + cols + self.target
            ]
            df_raw["date"] = pd.to_datetime(df_raw.date, format="%Y-%m-%d")
            df_raw["year"] = df_raw.date.dt.year
            df_raw["month"] = df_raw.date.dt.month
            groups = df_raw.groupby(["FOI_ID_LEVERANCIER"])
            # groups = df_raw.groupby(['FOI_ID_LEVERANCIER', 'year', 'month'])
            data_x = []
            data_y = []
            data_stamp = []
            targets_count = []
            seq_end_lengths = []
            # if self.early_classification != 12:
            #     print("early classification on")
            #     min_starting_date = datetime.strptime("2023-01", "%Y-%m")
            # else:
            min_starting_date = datetime.strptime("2023-03", "%Y-%m")
            self.dates = sorted(
                df_raw[df_raw["date"] > min_starting_date]["date"].unique()
            )
            # print(self.dates)
            # print(type(self.dates))

            # for 3 classes
            for (FOI_ID_LEVERANCIER), group_data in tqdm(groups):

                # skip 2 januari entries to make the length 81
                group_data = group_data[group_data["date"] > min_starting_date]
                # group_data = group_data[8:]
                month_data_x = group_data[cols].values.astype(np.float64)
                month_data_y = group_data[self.target].values.astype(
                    np.float64
                )
                if self.var_seq_len is True:
                    # create distribution of available timestamps, with mean at middle of the year
                    normal = torch.distributions.Normal(
                        len(group_data) / 2, len(group_data) / 4
                    )
                    seq_end = abs(int(normal.sample()))
                    if seq_end >= len(group_data):
                        seq_end = len(group_data) - 1
                    elif seq_end < 6:
                        seq_end = 6
                    seq_end_lengths.append(seq_end)
                else:
                    seq_end = len(group_data) - 1
                    seq_end_lengths.append(seq_end)

                if self.early_classification != 12:
                    # print("early classification on")
                    month = self.early_classification
                    date_string = "2023-{}".format(month)
                    end_date = datetime.strptime(date_string, "%Y-%m")
                    # Convert self.dates to a numpy array and ensure it's in datetime format
                    dates_array = np.array(self.dates, dtype="datetime64")

                    # Find the index of the last entry less than or equal to end_date
                    seq_end = np.searchsorted(
                        dates_array, end_date, side="right"
                    )

                target_values_flat = month_data_y.argmax(axis=1)
                targets_count.append(target_values_flat[0])
                data_x.append(month_data_x[0:seq_end])
                data_y.append(month_data_y[0:seq_end])


                df_stamp_month = group_data[["date"]][0:seq_end]
                df_stamp_month["date"] = pd.to_datetime(
                    df_stamp_month.date, format="%Y-%m-%d"
                )
                if self.timeenc == 0:
                    df_stamp_month["month"] = df_stamp_month.date.apply(
                        lambda row: row.month, axis=1
                    )
                    df_stamp_month["day"] = df_stamp_month.date.apply(
                        lambda row: row.day, axis=1
                    )
                    df_stamp_month["weekday"] = df_stamp_month.date.apply(
                        lambda row: row.weekday(), axis=1
                    )
                    df_stamp_month["hour"] = df_stamp_month.date.apply(
                        lambda row: row.hour, axis=1
                    )
                    data_stamp_month = df_stamp_month.drop(
                        ["date"], axis=1
                    ).values
                elif self.timeenc == 1:
                    data_stamp_month = time_features(
                        pd.to_datetime(df_stamp_month["date"].values),
                        freq=self.freq,
                    )
                    data_stamp_month = data_stamp_month.transpose(1, 0)
                data_stamp.append(data_stamp_month)

            max_length = int(max(map(len, data_x)))

            (data_x, data_y, data_stamp, masks) = pad_and_mask(
                data_x, data_y, data_stamp, max_length
            )
            num_train = int(len(data_x) * 0.7)
            num_test = int(len(data_x) * 0.1)
            num_vali = len(data_x) - num_train - num_test
            print(f"{num_train=}")

            # shuffle the data
            print(len(data_x), len(data_y), len(data_stamp))
            print(
                np.array(data_x).shape,
                np.array(data_y).shape,
                np.array(data_stamp).shape,
            )
            data_x, data_y, data_stamp, masks, seq_end_lengths = shuffle(
                np.array(data_x),
                np.array(data_y),
                np.array(data_stamp),
                np.array(masks),
                np.array(seq_end_lengths),
                random_state=5,
            )

            train_data_x = data_x[:num_train]
            print(f"{len(train_data_x)=}")
            test_data_x = data_x[num_train : num_train + num_test]
            vali_data_x = data_x[-num_vali:]
            total_data_x = [train_data_x, vali_data_x, test_data_x]

            train_data_y = data_y[:num_train]
            test_data_y = data_y[num_train : num_train + num_test]
            vali_data_y = data_y[-num_vali:]
            total_data_y = [train_data_y, vali_data_y, test_data_y]

            train_data_stamp = data_stamp[:num_train]
            test_data_stamp = data_stamp[num_train : num_train + num_test]
            vali_data_stamp = data_stamp[-num_vali:]
            total_data_stamp = [
                train_data_stamp,
                vali_data_stamp,
                test_data_stamp,
            ]

            train_masks = masks[:num_train]
            test_masks = masks[num_train : num_train + num_test]
            vali_masks = masks[-num_vali:]
            total_masks = [train_masks, vali_masks, test_masks]

            train_seq_end_lengths = seq_end_lengths[:num_train]
            test_seq_end_lengths = seq_end_lengths[
                num_train : num_train + num_test
            ]
            vali_seq_end_lengths = seq_end_lengths[-num_vali:]
            total_seq_end_lengths = [
                train_seq_end_lengths,
                vali_seq_end_lengths,
                test_seq_end_lengths,
            ]

            targets_count = sorted(targets_count)
            self.class_weights = compute_class_weight(
                class_weight="balanced",
                classes=np.unique(targets_count),
                y=targets_count,
            )
            if self.flag == "train":
                l = ["val", "test", "train"]
            elif self.flag == "val":
                l = ["train", "test", "val"]
            elif self.flag == "test":
                l = ["train", "val", "test"]
            for flag in l:
                self.split_id = self.type_map[flag]
                type_sort = type_map[self.split_id]
                path = os.path.join(self.root_path, "Netherlands/")
                folder = os.path.join(path, type_sort)
                if not os.path.exists(folder):
                    os.makedirs(folder)
                self.data_x = total_data_x[self.split_id]
                print(f"{len(self.data_x)=}")
                self.data_y = total_data_y[self.split_id]
                self.data_stamp = total_data_stamp[self.split_id]
                self.masks = total_masks[self.split_id]
                self.seq_end_lengths = total_seq_end_lengths[self.split_id]
                self.columns = cols
                np.save(os.path.join(folder, "columns.npy"), self.columns)
                # save these numpy arrays to a file
                np.save(
                    os.path.join(folder, "class_weights.npy"),
                    self.class_weights,
                )
                np.save(os.path.join(folder, "data_x.npy"), self.data_x)
                np.save(os.path.join(folder, "data_y.npy"), self.data_y)
                np.save(
                    os.path.join(folder, "data_stamp.npy"), self.data_stamp
                )
                np.save(os.path.join(folder, "masks.npy"), self.masks)

                np.save(
                    os.path.join(folder, "seq_end_lengths.npy"),
                    self.seq_end_lengths,
                )
                np.save(os.path.join(folder, "dates.npy"), self.dates)

    def __getitem__(self, index):
        seq_x = self.data_x[index].copy()
        target = self.data_y[index][0].copy()
        seq_x_mark = self.data_stamp[index].copy()
        seq_y_mark = self.data_stamp[index].copy()
        masks = self.masks[index].copy()
        seq_end_length = self.seq_end_lengths[index]
        return seq_x, target, seq_x_mark, seq_y_mark, masks, seq_end_length

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_dates(self):
        return self.dates

    def get_class_weights(self):
        return self.class_weights


def pad_and_mask(sequences, labels, data_stamp, max_length, padding_value=0.0):
    """
    Pads sequences, labels, and data_stamp to a maximum length and creates masks.

    Args:
        sequences (list): List of sequences.
        labels (list): List of labels.
        data_stamp (list): List of data stamps.
        max_length (int): Maximum length to pad sequences, labels, and data_stamp to.
        padding_value (float, optional): Value used for padding. Defaults to 0.0.

    Returns:
        tuple: A tuple containing the padded sequences, padded labels, padded data_stamp, and masks.
    """

    sequences_padded = (
        [
            pad(
                torch.tensor(s),
                (0, 0, 0, max(0, max_length - s.shape[0])),
                value=padding_value,
            )
            for s in sequences
        ],
    )

    labels_padded = (
        [
            pad(
                torch.tensor(l),
                (0, 0, 0, max(0, max_length - l.shape[0])),
                value=padding_value,
            )
            for l in labels
        ],
    )

    data_stamp = (
        [
            pad(
                torch.tensor(x),
                (0, 0, 0, max(0, max_length - x.shape[0])),
                value=padding_value,
            )
            for x in data_stamp
        ],
    )

    masks = []
    for s in sequences_padded:
        for t in s:
            # print(t)
            mask = t != padding_value
            masks.append(mask)


    return (sequences_padded[0], labels_padded[0], data_stamp[0], masks)
