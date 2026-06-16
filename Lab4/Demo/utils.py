import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def ValidCropResize(data_numpy, valid_frame_num, p_interval, window):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    begin = 0
    end = valid_frame_num
    valid_size = end - begin

    # crop
    if len(p_interval) == 1:
        p = p_interval[0]
        bias = int((1 - p) * valid_size / 2)
        data = data_numpy[:, begin + bias : end - bias, :, :]  # center_crop
        cropped_length = data.shape[1]
    else:
        p = np.random.rand(1) * (p_interval[1] - p_interval[0]) + p_interval[0]
        cropped_length = np.minimum(
            np.maximum(int(np.floor(valid_size * p)), 64), valid_size
        )  # constraint cropped_length lower bound as 64
        bias = np.random.randint(0, valid_size - cropped_length + 1)
        data = data_numpy[:, begin + bias : begin + bias + cropped_length, :, :]
        if data.shape[1] == 0:
            print(cropped_length, bias, valid_size)

    # resize
    data = torch.tensor(data, dtype=torch.float)
    data = data.permute(0, 2, 3, 1).contiguous().view(C * V * M, cropped_length)
    data = data[None, None, :, :]
    data = F.interpolate(
        data, size=(C * V * M, window), mode="bilinear", align_corners=False
    ).squeeze()  # could perform both up sample and down sample
    data = (
        data.contiguous().view(C, V, M, window).permute(0, 3, 1, 2).contiguous().numpy()
    )

    return data


def logistic_position_format(P, L=0.01):
    """
    使用 Logistic 位置格式将姿势数据映射到 0-255 的区间
    :param P: 归一化后的相对位置张量
    :param L: Logistic 位置格式中的常数
    :return: 映射后的张量
    """
    return np.ceil(255 / (1 + np.exp(-L * P)))


def body_centered_normalization(data):
    """
    身体中心归一化，将第一帧中的 spinebase 作为局部坐标系的原点
    :param data: 形状为 (coordinates, frames, markers) 的关节位置张量
    :return: 归一化后的张量
    """
    # spinebase = data[:, 0, 0].unsqueeze(1).unsqueeze(2)  # 获取第一帧的 spinebase 位置
    spinebase = data[:, 0, 0]  # 获取第一帧的 spinebase 位置
    spinebase = spinebase[:, np.newaxis, np.newaxis]
    return data - spinebase


# 转换数据为RGB图像
def ConvertToRGB(data):
    """
    构建 3D 姿势图像
    :param data: 形状为 (coordinates, frames, markers) 的关节位置张量
    :return: 形状为 (frames, markers, 3) 的 3D 姿势图像张量
    """
    data = body_centered_normalization(data)
    rgb_image = logistic_position_format(data)
    # rgb_image = rgb_image.transpose(1, 2, 0)  # 转换形状为 (frames, markers, 3)
    # return rgb_image.astype(np.uint8)
    return rgb_image


def preprocess_data(dir_path, split_idx=0):
    dir_list = os.listdir(dir_path)

    joint_index = [
        6,
        7,
        8,
        13,
        14,
        15,
        20,
        21,
        22,
        27,
        28,
        29,
        34,
        35,
        36,
        41,
        42,
        43,
        48,
        49,
        50,
        55,
        56,
        57,
        62,
        63,
        64,
        174,
        175,
        176,
        181,
        182,
        183,
        188,
        189,
        190,
        195,
        196,
        197,
        307,
        308,
        309,
        314,
        315,
        316,
        321,
        322,
        323,
        335,
        336,
        337,
        342,
        343,
        344,
        349,
        350,
        351,
    ]
    dataset, subject, action = [], [], []
    for file in dir_list:
        if (
            file.startswith("subject000")
            or file.startswith("subject030")
            or file.startswith("subject031")
        ):
            continue
        raw_data = pd.read_csv(dir_path + "/" + file, skiprows=3)
        frames = int(raw_data.iloc[-1, 0])
        data = []
        try:
            for f in range(3, frames + 4, 4):
                line = raw_data.values[f, [j for j in joint_index]]
                line = line.reshape(19, 3).astype(np.float32)
                data.append(line)
        except:
            # 写入txt文件里
            with open("./error.txt", "a") as f:
                f.write(file + "\n")
            continue
        data = np.stack(data)
        data = data[:, :, :, np.newaxis]
        data = data.transpose(2, 0, 1, 3)
        valid_frame_num = np.sum(data.sum(0).sum(-1).sum(-1) != 0)
        data = ValidCropResize(data, valid_frame_num, [0.95], 64)[:, :, :, 0]
        rgb_data = ConvertToRGB(data)

        dataset.append(rgb_data)
        subject.append(file[7:10])
        action.append(file[17:20])

    dataset = np.stack(dataset)
    subject = np.stack(subject).astype(int) - 1
    action = np.stack(action).astype(int) - 1
    dataset = torch.from_numpy(dataset)
    action = torch.from_numpy(action)

    # 使用 train_test_split 进行分层抽样
    X_train, X_test, y_train, y_test = train_test_split(
        dataset, action, test_size=0.2, stratify=action, random_state=42
    )

    return 8, y_train, y_test, X_train, X_test


def accuracy(logits, labels):
    _, indices = torch.max(logits, dim=1)
    correct = torch.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)
