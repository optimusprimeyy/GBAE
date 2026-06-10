import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import os
from GBshengcheng_v2 import getGranularBall  # 假设这个模块存在


# --- EarlyStopping 类（适配无验证集：监控训练损失）---
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


# --- 粒球类 ---
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


# --- 双分支解耦AE模型（新增：结构对齐 + 轻量互信息模块）---
class DisentangledGBAE(nn.Module):
    def __init__(self, input_dim, latent_dim=16):
        super().__init__()
        # 新增：global分支增强（结构对齐），多加一层保证表达能力
        self.encoder_local = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim // 2, latent_dim)
        )
        self.decoder_local = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim // 2, input_dim)
        )
        self.encoder_global = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim // 2, input_dim // 4),  # 新增：global分支多一层
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim // 4, latent_dim)
        )
        self.decoder_global = nn.Sequential(
            nn.Linear(latent_dim, input_dim // 4),  # 新增：对应encoder的结构
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim // 4, input_dim // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(input_dim // 2, input_dim)
        )

    def forward(self, x, centers):
        z_local = self.encoder_local(x)
        z_global = self.encoder_global(centers)
        recon_local = self.decoder_local(z_local)
        recon_global = self.decoder_global(z_global)
        return recon_local, recon_global, z_local, z_global

    # 新增：轻量互信息损失（InfoNCE变体，计算快、效果好）
    def compute_mi_loss(self, z_local, z_global, temperature=0.1):
        # 归一化潜变量（不增加计算量）
        z_l = F.normalize(z_local, dim=1)
        z_g = F.normalize(z_global, dim=1)
        # 计算相似度矩阵（矩阵乘法是GPU原生加速，速度快）
        sim_matrix = torch.mm(z_l, z_g.T) / temperature
        # 正样本：对角线（同一样本的z_l和z_g）
        pos_sim = torch.diag(sim_matrix)
        # 负样本：非对角线元素（屏蔽自身）
        neg_sim = sim_matrix - torch.eye(sim_matrix.shape[0], device=sim_matrix.device) * 1e9
        # InfoNCE损失（最小化互信息，仅几行计算）
        mi_loss = -pos_sim + torch.logsumexp(neg_sim, dim=1)
        return mi_loss.mean()


# --- 辅助函数：稳健归一化（修改：增加NaN/Inf保护）---
def robust_scale(s):
    median = np.median(s)
    iqr = np.percentile(s, 75) - np.percentile(s, 25)
    # 修改：防止分母为0，兜底逻辑
    if abs(iqr) < 1e-10:
        std = np.std(s)
        if abs(std) < 1e-10:
            return np.zeros_like(s)
        return (s - median) / (std + 1e-10)
    return (s - median) / (iqr + 1e-10)


# --- 核心修改：训练函数（集成互信息损失，保证速度）---
def train_disentangled_model(center_data, sample_to_gb_idx, x_train, epochs=200, batch_size=32, patience=15,
                             lambda_de=0.5, delta=1e-6, save_folder=None, data_name=None):  # 修改：lambda_de默认0.5增强解耦
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = x_train.shape[1]
    sample_centers = center_data[sample_to_gb_idx]

    x_tensor = torch.tensor(x_train, dtype=torch.float32).to(device)
    c_tensor = torch.tensor(sample_centers, dtype=torch.float32).to(device)

    model = DisentangledGBAE(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    early_stopping = EarlyStopping(patience=patience, verbose=False)

    epoch_bar = tqdm(
        range(epochs),
        desc=f"delta={delta}  Train λ={lambda_de}",
        unit="ep",
        ncols=100,
        leave=True,
        bar_format="{l_bar}{bar:30}{r_bar}"
    )

    best_train_loss = float('inf')
    best_model = None

    for epoch in epoch_bar:
        model.train()
        permutation = torch.randperm(len(x_tensor))
        epoch_loss = 0.0

        # 批次训练（保持原有批次逻辑，不增加耗时）
        for i in range(0, len(x_tensor), batch_size):
            indices = permutation[i:i + batch_size]
            batch_x = x_tensor[indices]
            batch_c = c_tensor[indices]

            optimizer.zero_grad()
            recon_l, recon_g, z_l, z_g = model(batch_x, batch_c)

            # 1. 重构损失（不变）
            loss_recon = F.mse_loss(recon_l, batch_x) + F.mse_loss(recon_g, batch_c)
            # 2. 解耦损失（修改：正交约束 + 互信息损失，轻量组合）
            z_l_norm = F.normalize(z_l, p=2, dim=1)
            z_g_norm = F.normalize(z_g, p=2, dim=1)
            # 正交约束（原有逻辑，无新增计算）
            cos_sim = torch.sum(z_l_norm * z_g_norm, dim=1)
            ortho_loss = torch.mean(cos_sim ** 2)
            # 互信息损失（新增：轻量计算，GPU加速）
            mi_loss = model.compute_mi_loss(z_l, z_g)
            # 组合解耦损失（权重可调，保证总计算量不变）
            loss_decouple = ortho_loss + 0.1 * mi_loss
            # 3. 总损失
            total_loss = loss_recon + lambda_de * loss_decouple

            total_loss.backward()
            optimizer.step()
            epoch_loss += total_loss.item() * len(batch_x)

        # 计算平均训练损失
        epoch_loss /= len(x_tensor)

        # 保存最优模型（基于训练损失）
        if epoch_loss < best_train_loss:
            best_train_loss = epoch_loss
            best_model = model.state_dict()

        # 早停检查（监控训练损失）
        early_stopping(epoch_loss)
        epoch_bar.set_postfix({"Train Loss": f"{epoch_loss:.4f}"})

        if early_stopping.early_stop:
            tqdm.write(f"✅ 早停触发！停止于 Epoch {epoch + 1}")
            break

    epoch_bar.close()

    # 加载最优模型并保存
    model.load_state_dict(best_model)
    if save_folder and data_name:
        model_path = os.path.join(save_folder, f"GBAE_{data_name}_delta-{delta}_lambda-{lambda_de}.pth")
        torch.save(model.state_dict(), model_path)

    return model


# --- 评分函数（修改：全面NaN/Inf保护，保证输出干净）---
def compute_disentangled_scores(x_test, center_data_train, sample_to_gb_idx_train, model):
    device = next(model.parameters()).device

    # 为测试集样本匹配训练集的粒球中心（核心：保持粒球分布一致）
    sample_centers_test = []
    for x in x_test:
        # 找距离最近的训练集粒球中心
        distances = np.sqrt(np.sum((center_data_train - x) ** 2, axis=1))
        closest_gb_idx = np.argmin(distances)
        sample_centers_test.append(center_data_train[closest_gb_idx])
    sample_centers_test = np.array(sample_centers_test)

    x_t = torch.tensor(x_test, dtype=torch.float32).to(device)
    c_t = torch.tensor(sample_centers_test, dtype=torch.float32).to(device)

    model.eval()
    with torch.no_grad():
        recon_l, recon_g, z_l, z_g = model(x_t, c_t)
        # 修改：限制重构误差范围，防止Inf
        score_r = torch.mean((recon_l - x_t) ** 2, dim=1)
        score_r = torch.clamp(score_r, 0, 1e6).cpu().numpy()

        dist_orig = np.sqrt(np.sum((x_test - sample_centers_test) ** 2, axis=1))
        # 修改：限制潜空间距离范围
        dist_latent = torch.norm(z_l - z_g, p=2, dim=1)
        dist_latent = torch.clamp(dist_latent, 0, 1e6).cpu().numpy()

        score_c = np.abs(dist_orig.flatten() - dist_latent)
        # 修改：方差加最小值限制，防止分母为0
        var_l_per_sample = torch.var(z_l, dim=1)
        var_l_per_sample = torch.clamp(var_l_per_sample, 1e-10, 1e6).cpu().numpy()
        var_g_per_sample = torch.var(z_g, dim=1)
        var_g_per_sample = torch.clamp(var_g_per_sample, 1e-10, 1e6).cpu().numpy()

        # 修改：分母加大安全值，NaN兜底
        alpha = var_l_per_sample / (var_l_per_sample + var_g_per_sample + 1e-6)
        alpha = np.where(np.isnan(alpha), 0.5, alpha)  # NaN替换为0.5
        alpha = np.clip(alpha, 0.0, 1.0)  # 限制0-1

    s_r_norm = robust_scale(score_r)
    s_c_norm = robust_scale(score_c)
    final_score = alpha * s_r_norm + (1 - alpha) * s_c_norm

    # 修改：最后一道兜底，确保无NaN/Inf
    final_score = np.where(np.isnan(final_score), 0.0, final_score)
    final_score = np.where(np.isinf(final_score), 0.0, final_score)

    return final_score, s_r_norm, s_c_norm


# --- add_center函数 ---
def add_center(gb_list, n):
    gb_dist = []
    center_data = []
    sample_to_gb_idx = np.full((n,), -1, dtype=int)

    for gb_idx in range(len(gb_list)):
        gb = GranularBall(gb_list[gb_idx], gb_idx)
        gb_dist.append(gb)
        center_data.append(gb.center)
        for sample_idx in gb.sample_indices:
            if sample_idx < n:  # 防止索引越界
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
    model = train_disentangled_model(
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
    final_score, Score_R, Score_C = compute_disentangled_scores(
        x_test=X_test,
        center_data_train=center_data_train,
        sample_to_gb_idx_train=sample_to_gb_idx_train,
        model=model
    )

    return final_score


# --- 测试代码 --