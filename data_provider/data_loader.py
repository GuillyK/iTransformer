import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from utils.timefeatures import time_features
import warnings

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
        self.type_map = {"train": 0, "val": 1, "test": 2}
        self.split_id = self.type_map[flag]

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
        type_map = ["train", "val", "test"]
        type_sort = type_map[self.split_id]
        dataset_files = ["data_x.npy", "data_y.npy", "data_stamp.npy"]
        path = os.path.join(self.root_path ,"noordoostpolder/")
        folder = os.path.join(path, type_sort)
        # check if in path train/test/val folder exists if not create
        if os.path.exists(folder):
            self.data_x = np.load(os.path.join(folder, "data_x.npy"))
            self.data_y = np.load(os.path.join(folder, "data_y.npy"))
            self.data_stamp = np.load(os.path.join(folder, "data_stamp.npy"))
        else:
            os.makedirs(folder)

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
            df_raw[cols]=scaled_data
            # groups = df_raw.groupby("FOI_ID_LEVERANCIER")
            # print("these are the groups\n", groups)
            df_raw = df_raw[
                ["date"] + ["FOI_ID_LEVERANCIER"] + cols + self.target
            ]
            df_raw["date"] = pd.to_datetime(df_raw.date)
            df_raw['year'] = df_raw.date.dt.year
            df_raw['month'] = df_raw.date.dt.month
            groups = df_raw.groupby(['FOI_ID_LEVERANCIER', 'year', 'month'])
            data_x = []
            data_y = []
            data_stamp = []
            desired_length = 30 #seq_length maybe later
            # Loop over the groups
            i = 0
            for (FOI_ID_LEVERANCIER, year, month), group_data in tqdm(groups):
                # Now, group_data contains the data for one 'FOI_ID_LEVERANCIER' for one month
                # i+=1
                # if i > 200:
                #     break
                # skip januari for now since it has 16 days #TODO: fix this
                if month == 1:
                    continue
                month_data_x = group_data[cols].values.astype(np.float64)
                month_data_y = group_data[self.target].values.astype(np.float64)
                if len(month_data_x) < desired_length:
                    padding = desired_length - len(month_data_x)
                    month_data_x = np.pad(month_data_x, ((0, padding), (0, 0)), mode='constant')
                    month_data_y = np.pad(month_data_y, ((0, padding), (0, 0)), mode='constant')
                elif len(month_data_x) > desired_length:
                    month_data_x = month_data_x[:desired_length]
                    month_data_y = month_data_y[:desired_length]


                df_stamp_month = group_data[["date"]]
                df_stamp_month["date"] = pd.to_datetime(df_stamp_month.date)
                if self.timeenc == 0:
                    df_stamp_month["month"] = df_stamp_month.date.apply(
                        lambda row: row.month, axis=1
                    )
                    df_stamp_month["day"] = df_stamp_month.date.apply(lambda row: row.day, axis=1)
                    df_stamp_month["weekday"] = df_stamp_month.date.apply(
                        lambda row: row.weekday(), axis=1
                    )
                    df_stamp_month["hour"] = df_stamp_month.date.apply(
                        lambda row: row.hour, axis=1
                    )
                    data_stamp_month = df_stamp_month.drop(["date"], axis=1).values
                elif self.timeenc == 1:
                    data_stamp_month = time_features(
                        pd.to_datetime(df_stamp_month["date"].values), freq=self.freq
                    )
                    data_stamp_month = data_stamp_month.transpose(1, 0)
                # Pad data_stamp_month to a size of 30
                if len(data_stamp_month) < desired_length:
                    padding = desired_length - len(data_stamp_month)
                    data_stamp_month = np.pad(data_stamp_month, ((0, padding), (0, 0)), mode='constant')
                elif len(data_stamp_month) > desired_length:
                    data_stamp_month = data_stamp_month[:desired_length]

                data_x.append(month_data_x)
                data_y.append(month_data_y)
                data_stamp.append(data_stamp_month)



            num_train = int(len(data_x) * 0.7)
            num_test = int(len(data_x) * 0.1)
            num_vali = len(data_x) - num_train - num_test

            train_data_x = data_x[:num_train]
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
            total_data_stamp = [train_data_stamp, vali_data_stamp, test_data_stamp]

            self.data_x = np.array(total_data_x[self.split_id])
            self.data_y = np.array(total_data_y[self.split_id])
            self.data_stamp = np.array(total_data_stamp[self.split_id])
            # save these numpy arrays to a file
            np.save(os.path.join(folder,"data_x.npy") , self.data_x)
            np.save(os.path.join(folder,"data_y.npy"), self.data_y)
            np.save(os.path.join(folder,"data_stamp.npy"), self.data_stamp)



        # Borders to later get the the number of groups needed for training, validation, and testing
        # border1s = [
        #     0,
        #     num_train - self.seq_len,
        #     len(groups) - num_test - self.seq_len,
        # ]
        # border2s = [num_train, num_train + num_vali, len(groups)]
        # # Border split for train, val, and test
        # border1 = border1s[self.split_id]
        # border2 = border2s[self.split_id]

        # Create a list of group keys
        # group_keys = groups.groups.keys()

        # # Split the keys into training, validation, and testing keys
        # train_keys = list(group_keys)[:num_train]
        # test_keys = list(group_keys)[num_train : num_train + num_test]
        # vali_keys = list(group_keys)[-num_vali:]

        # # Create training, validation, and testing groups
        # train_groups = groups.filter(lambda x: x.name in train_keys)
        # test_groups = groups.filter(lambda x: x.name in test_keys)
        # vali_groups = groups.filter(lambda x: x.name in vali_keys)
        # list_of_groups = [train_groups, vali_groups, test_groups]

        # amount_features = len(cols)
        # cols_data = df_raw.columns[1 : amount_features + 1]
        # cur_group = list_of_groups[self.split_id]
        # df_data = cur_group[cols_data]


        # df_stamp = cur_group[["date"]]
        # df_stamp["date"] = pd.to_datetime(df_stamp.date)
        # if self.timeenc == 0:
        #     df_stamp["month"] = df_stamp.date.apply(
        #         lambda row: row.month, axis=1
        #     )
        #     df_stamp["day"] = df_stamp.date.apply(lambda row: row.day, axis=1)
        #     df_stamp["weekday"] = df_stamp.date.apply(
        #         lambda row: row.weekday(), axis=1
        #     )
        #     df_stamp["hour"] = df_stamp.date.apply(
        #         lambda row: row.hour, axis=1
        #     )
        #     data_stamp = df_stamp.drop(["date"], axis=1).values
        # elif self.timeenc == 1:
        #     data_stamp = time_features(
        #         pd.to_datetime(df_stamp["date"].values), freq=self.freq
        #     )
        #     data_stamp = data_stamp.transpose(1, 0)

        # self.data_x = cur_group[cols_data].values.astype(np.float64)
        # self.data_y = cur_group[self.target].values.astype(np.float64)
        # self.data_stamp = data_stamp_month

        # print("These are the target values", self.target, cur_group[self.target])

    def __getitem__(self, index):
        seq_x = self.data_x[index].copy()
        target = self.data_y[index][0].copy()
        seq_x_mark = self.data_stamp[index].copy()
        seq_y_mark = self.data_stamp[index].copy()
        # day day day
        # jan, feb, mar, april, may, june, july, aug, sept, oct, nov, dec, field2 jan, field2 feb
        #

        # s_begin = index
        # s_end = s_begin + self.seq_len
        # r_begin = s_end - self.label_len
        # r_end = r_begin + self.label_len + self.pred_len

        # seq_x = self.data_x[s_begin:s_end]
        # seq_y = self.data_y[r_begin:r_end]
        # seq_x_mark = self.data_stamp[s_begin:s_end]
        # seq_y_mark = self.data_stamp[index]
        # seq_y = self.data_y[index]
        # print(seq_y, self.data_y, self.data_y.shape, self.data_x.shape)
        # exit()
        return seq_x, target, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
