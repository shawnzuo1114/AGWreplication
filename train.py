import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import random
from torchvision import transforms

# -----------------------------
# 数据集处理
# -----------------------------
data_path = '../Datasets/SYSU-MM01/ori_data'
fix_image_width, fix_image_height = 144, 288
rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras  = ['cam3','cam6']

# 读取训练 ID
with open(os.path.join(data_path,'exp/train_id.txt'),'r') as f:
    ids_train = [int(x) for x in f.read().splitlines()[0].split(',')]
with open(os.path.join(data_path,'exp/val_id.txt'),'r') as f:
    ids_val = [int(x) for x in f.read().splitlines()[0].split(',')]
id_train = ["%04d" % x for x in (ids_train + ids_val)]

def collect_images(id_list, cameras):
    files = []
    for pid in sorted(id_list):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                files.extend(sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir)]))
    return files

files_rgb = collect_images(id_train, rgb_cameras)
files_ir  = collect_images(id_train, ir_cameras)

pid_container = set(int(p[-13:-9]) for p in files_ir)
pid2label = {pid:label for label, pid in enumerate(pid_container)}

def read_imgs(file_list):
    imgs, labels = [], []
    for path in file_list:
        img = Image.open(path).resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        imgs.append(np.array(img))
        pid = int(path[-13:-9])
        labels.append(pid2label[pid])
    return np.array(imgs), np.array(labels)

train_rgb_img, train_rgb_label = read_imgs(files_rgb)
train_ir_img, train_ir_label   = read_imgs(files_ir)

# -----------------------------
# Dataset & DataLoader
# -----------------------------
class SYSUDataset(Dataset):
    def __init__(self, rgb_imgs, rgb_labels, ir_imgs, ir_labels, transform=None):
        self.rgb_imgs = rgb_imgs
        self.rgb_labels = rgb_labels
        self.ir_imgs = ir_imgs
        self.ir_labels = ir_labels
        self.transform = transform

    def __getitem__(self, index):
        img_rgb = self.rgb_imgs[index]
        img_ir  = self.ir_imgs[index]
        label_rgb = self.rgb_labels[index]
        label_ir  = self.ir_labels[index]

        if self.transform:
            img_rgb = self.transform(img_rgb)
            img_ir  = self.transform(img_ir)
        return img_rgb, img_ir, label_rgb, label_ir

    def __len__(self):
        return len(self.rgb_labels)

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5],[0.5])
])

train_dataset = SYSUDataset(train_rgb_img, train_rgb_label, train_ir_img, train_ir_label, transform_train)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)

# -----------------------------
# AGW 模型定义 (假设你已有实现)
# -----------------------------
from model import agw, Bottleneck

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = agw(Bottleneck, [3,4,6,3]).to(device)

# -----------------------------
# 损失函数
# -----------------------------
from loss import IDLoss, TripletLoss_WRT
num_classes = len(pid2label)
criterion_id  = IDLoss(num_classes=num_classes).to(device)
criterion_wrt = TripletLoss_WRT().to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# -----------------------------
# 训练函数
# -----------------------------
def train_one_epoch(model, loader, optimizer, criterion_id, criterion_wrt):
    model.train()
    total_loss = 0
    total_correct = 0
    for imgs_rgb, imgs_ir, labels_rgb, labels_ir in loader:
        imgs_rgb = imgs_rgb.to(device, dtype=torch.float)
        imgs_ir  = imgs_ir.to(device, dtype=torch.float)
        labels_rgb = labels_rgb.to(device)
        labels_ir  = labels_ir.to(device)

        optimizer.zero_grad()
        feat_rgb = model(imgs_rgb)
        feat_ir  = model(imgs_ir)

        loss_id_rgb = criterion_id(feat_rgb, labels_rgb)
        loss_id_ir  = criterion_id(feat_ir, labels_ir)
        loss_wrt, correct = criterion_wrt(feat_rgb, feat_ir, labels_rgb)

        loss = loss_id_rgb + loss_id_ir + loss_wrt
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_correct += correct
    return total_loss / len(loader), total_correct

# -----------------------------
# 特征提取 & 评价指标
# -----------------------------
def extract_features(model, loader):
    model.eval()
    feats, labels = [], []
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device, dtype=torch.float)
            feat = model(imgs)
            feats.append(feat.cpu())
            labels.append(lbls)
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    return feats, labels

def evaluate(feat_q, label_q, feat_g, label_g):
    # 计算距离矩阵
    distmat = torch.cdist(feat_q, feat_g)
    m, n = distmat.shape
    dist_np = distmat.numpy()
    rank1 = rank5 = rank10 = rank20 = mINP = 0

    correct_count = np.zeros(20)
    INP_total = []
    for i in range(m):
        q_label = label_q[i].item()
        g_labels = label_g.numpy()
        sorted_indices = np.argsort(dist_np[i])
        match = (g_labels[sorted_indices] == q_label).astype(int)
        # rank-k
        for k in [1,5,10,20]:
            if match[:k].sum() > 0:
                if k==1: rank1+=1
                elif k==5: rank5+=1
                elif k==10: rank10+=1
                elif k==20: rank20+=1
        # mINP
        indices = np.where(match==1)[0]
        if len(indices) > 0:
            INP = 1.0 / (indices[-1]+1)
            INP_total.append(INP)
    mINP = np.mean(INP_total)

    return rank1/m, rank5/m, rank10/m, rank20/m, mINP

# -----------------------------
# 模拟 Query/Gallery Dataset
# -----------------------------
class TestDataset(Dataset):
    def __init__(self, imgs, labels, transform=None):
        self.imgs = imgs
        self.labels = labels
        self.transform = transform

    def __getitem__(self, index):
        img, lbl = self.imgs[index], self.labels[index]
        if self.transform:
            img = self.transform(img)
        return img, lbl

    def __len__(self):
        return len(self.labels)

# 使用训练集的一部分作为 query/gallery 示例
query_loader   = DataLoader(TestDataset(train_rgb_img[:100], train_rgb_label[:100], transform_train), batch_size=16)
gallery_loader = DataLoader(TestDataset(train_ir_img[:100], train_ir_label[:100], transform_train), batch_size=16)

# -----------------------------
# 主训练循环
# -----------------------------
num_epochs = 30
for epoch in range(num_epochs):
    loss, correct = train_one_epoch(model, train_loader, optimizer, criterion_id, criterion_wrt)
    print(f"Epoch [{epoch+1}/{num_epochs}] Loss:{loss:.4f} Correct:{correct}")

    # 提取特征并评估
    feat_q, label_q = extract_features(model, query_loader)
    feat_g, label_g = extract_features(model, gallery_loader)
    r1,r5,r10,r20,mINP_val = evaluate(feat_q, label_q, feat_g, label_g)
    print(f"Rank-1:{r1:.4f} Rank-5:{r5:.4f} Rank-10:{r10:.4f} Rank-20:{r20:.4f} mINP:{mINP_val:.4f}")
