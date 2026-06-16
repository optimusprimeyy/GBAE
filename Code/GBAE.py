import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from GBshengcheng_v2 import getGranularBall  # 假设这个模块存在


# --- Class EarlyStopping---
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=1e-6):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, train_loss):
        score = -train_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


# --- class GranularBall ---
class GranularBall:
    def __init__(self, data, index):
        self.data = data[:, :-1]
        self.index = index
        self.center = self.data.mean(0)
        self.score = 0
        self.radius = self.calculate_radius()
        self.sample_indices = data[:, -1].astype(int)

    def calculate_radius(self):
        distances = np.sqrt(np.sum((self.data - self.center) ** 2, axis=1))
        radius = np.max(distances) if len(self.data) > 1 else 1e-6
        return radius


def robust_scale(s):
    median = np.median(s)
    iqr = np.percentile(s, 75) - np.percentile(s, 25)
    # Edit: prevent division by zero, add fallback logic
    if abs(iqr) < 1e-10:
        std = np.std(s)
        if abs(std) < 1e-10:
            return np.zeros_like(s)
        return (s - median) / (std + 1e-10)
    return (s - median) / (iqr + 1e-10)



# --- add_center ---
def add_center(gb_list, n):
    gb_dist = []
    center_data = []
    sample_to_gb_idx = np.full((n,), -1, dtype=int)

    for gb_idx in range(len(gb_list)):
        gb = GranularBall(gb_list[gb_idx], gb_idx)
        gb_dist.append(gb)
        center_data.append(gb.center)
        for sample_idx in gb.sample_indices:
            if sample_idx < n:
                sample_to_gb_idx[sample_idx] = gb_idx

    center_data = np.array(center_data)
    return gb_dist, center_data, sample_to_gb_idx


# --- 核心修改：GB_AE函数（接收训练/测试集，返回测试集评分）---
def GB_AE(X_train, X_test, delta, lambda_de, save_folder, data_name):
    # 仅基于训练集生成粒球
    gb_list_raw = getGranularBall(X_train, delta)
    n = len(X_train)
    gb_list, center_data_train, sample_to_gb_idx_train = add_center(gb_list_raw, n)

    # 训练模型（保存最优模型）
    model = GB_AE(
        center_data=center_data_train,
        sample_to_gb_idx=sample_to_gb_idx_train,
        x_train=X_train,
        epochs=200,
        batch_size=32,
        lambda_de=lambda_de,
        delta=delta,
        save_folder=save_folder,
        data_name=data_name
    )

    # 计算测试集评分
    final_score, Score_R, Score_C = score(
        x_test=X_test,
        center_data_train=center_data_train,
        sample_to_gb_idx_train=sample_to_gb_idx_train,
        model=model
    )

    return final_score

