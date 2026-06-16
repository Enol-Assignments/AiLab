# Dataset.py
import sys
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def ValidCropResize(data_numpy, valid_frame_num, p_interval, window):
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias : end - bias, :, :]
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), 64), valid_size
        )
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias : begin + bias + cropped_length, :, :]

    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(
        data, size=(C * V * M, window), mode="bilinear", align_corners=False
    ).squeeze()
    data = (
        data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
    )
    return data


def NormalizeCenteredBody(data):
    spinebase = data[:, 0, 0]
    spinebase = spinebase[:, np.newaxis, np.newaxis]
    return data - spinebase


def ConvertToRGB(data):
    data = NormalizeCenteredBody(data)
    rgb_image = np.ceil(255 / (1 + np.exp(-0.01 * data)))
    return rgb_image


def LoadAndPreprocessData(
    datasets_path="Datasets/",datacleaned_file="DataCleaned.pt"
):
    dir_list = os.listdir(datasets_path)
    joint_index = [
        6, 7, 8, 13, 14, 15,
        20, 21, 22, 27, 28, 29,
        34, 35, 36,
        41, 42, 43, 48, 49,
        50, 55, 56, 57,
        62, 63, 64,
        174, 175, 176,
        181, 182, 183, 188, 189,
        190, 195, 196, 197,
        307, 308, 309,
        314, 315, 316,
        321, 322, 323,
        335, 336, 337,
        342, 343,
        344, 349,
        350, 351,
    ]
    dataset, action = [], []

    print("开始清洗并加载骨骼点数据集……")

    for file in dir_list:
        if (
            file.startswith("subject000")
            or file.startswith("subject030")
            or file.startswith("subject031")
        ):
            continue

        sys.stdout.write("正在处理第 %d/%d 个文件：%s\r" % (len(dataset) + 1, len(dir_list), file))

        raw_data = pd.read_csv(
            os.path.join(datasets_path, file), skiprows=3, low_memory=False
        )
        frames = int(raw_data.iloc[-1, 0])
        data = []
        try:
            for f in range(3, frames + 4, 4):
                line = raw_data.values[f, [j for j in joint_index]]
                line = line.reshape(19, 3).astype(np.float32)
                data.append(line)
        except:
            with open("./Error.txt", "a") as e_f:
                e_f.write(file + "\n")
            continue

        data = np.stack(data)
        data = data[:, :, :, np.newaxis]
        data = data.transpose(2, 0, 1, 3)
        valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        data = ValidCropResize(data, valid_frame_num, [0.95], 64)[:, :, :, 0]
        rgb_data = ConvertToRGB(data)

        dataset.append(rgb_data)
        action.append(file[17:20])

    sys.stdout.flush()

    dataset = np.stack(dataset)
    action = np.stack(action).astype(int) - 1

    X_tensor = torch.from_numpy(dataset).float()
    y_tensor = torch.from_numpy(action).long()

    X_train, X_test, y_train, y_test = train_test_split(
        X_tensor, y_tensor, test_size=0.2, stratify=y_tensor, random_state=42
    )

    torch.save(
        {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test},
        datacleaned_file,
    )
