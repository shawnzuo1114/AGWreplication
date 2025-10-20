import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from model import agw, Bottleneck
from loss import AGW_Loss

# -----------------------------
# 配置
# -----------------------------
data_path = '../../dataset/SYSU-MM01'
fix_image_width, fix_image_height = 144, 288
rgb_cameras = ['cam1','cam2','cam4','cam5']
ir_cameras  = ['cam3','cam6']
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# -----------------------------
# 读取训练ID
# -----------------------------
with open(os.path.join(data_path,'exp/train_id.txt'),'r') as f:
    ids_train = [int(x) for x in f.read().splitlines()[0].split(',')]
with open(os.path.join(data_path,'exp/val_id.txt'),'r') as f:
    ids_val = [int(x) for x in f.read().splitlines()[0].split(',')]
id_train = ["%04d" % x for x in (ids_train + ids_val)]

def collect_images(id_list, cameras, max_num=None):
    files = []
    for pid in sorted(id_list):
        for cam in cameras:
            img_dir = os.path.join(data_path, cam, pid)
            if os.path.isdir(img_dir):
                img_paths = sorted([os.path.join(img_dir,f) for f in os.listdir(img_dir)])
                files.extend(img_paths)
                if max_num and len(files) >= max_num:
                    return files[:max_num]
    return files

def read_imgs(file_list):
    imgs, labels = [], []
    for path in file_list:
        img = Image.open(path).resize(
            (fix_image_width, fix_image_height),
            resample=Image.Resampling.LANCZOS
        )
        img = np.array(img)
        if img.ndim == 2:  # 单通道 IR 图像，复制到3通道
            img = np.stack([img]*3, axis=-1)
        imgs.append(img)
        pid = int(path[-13:-9])
        labels.append(pid2label[pid])
    return np.array(imgs), np.array(labels)

# -----------------------------
# Dataset 定义
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

transform_train = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

# -----------------------------
# 主程序
# -----------------------------
if __name__ == '__main__':
    # -----------------------------
    # 读取图片并生成标签
    # -----------------------------
    files_rgb = collect_images(id_train, rgb_cameras, max_num=200)
    files_ir  = collect_images(id_train, ir_cameras, max_num=200)

    pid_container = set(int(p[-13:-9]) for p in files_ir)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}

    train_rgb_img, train_rgb_label = read_imgs(files_rgb)
    train_ir_img, train_ir_label   = read_imgs(files_ir)

    # -----------------------------
    # DataLoader
    # -----------------------------
    train_dataset = SYSUDataset(train_rgb_img, train_rgb_label, train_ir_img, train_ir_label, transform_train)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)  # Windows调试用num_workers=0

    query_loader   = DataLoader(TestDataset(train_rgb_img[:100], train_rgb_label[:100], transform_train), batch_size=16)
    gallery_loader = DataLoader(TestDataset(train_ir_img[:100], train_ir_label[:100], transform_train), batch_size=16)

    # -----------------------------
    # 模型、分类头、损失、优化器
    # -----------------------------
    model = agw(Bottleneck, [3,4,6,3]).to(device)
    num_classes = len(pid2label)
    classifier = torch.nn.Linear(2048, num_classes).to(device)
    criterion = AGW_Loss(num_classes=num_classes, lambda_wrt=1.0)
    optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=3e-4)

    # -----------------------------
    # 训练函数
    # -----------------------------
    def train_one_epoch(model, loader, optimizer, criterion, classifier):
        model.train()
        total_loss = total_id_loss = total_wrt_loss = total_correct = 0
        for imgs_rgb, imgs_ir, labels_rgb, labels_ir in loader:
            imgs_rgb = imgs_rgb.to(device, dtype=torch.float)
            imgs_ir  = imgs_ir.to(device, dtype=torch.float)
            labels_rgb = labels_rgb.to(device, dtype=torch.long)
            labels_ir  = labels_ir.to(device, dtype=torch.long)


            optimizer.zero_grad()
            feat_rgb = model(imgs_rgb)
            feat_ir  = model(imgs_ir)

            loss_rgb, id_loss_rgb, wrt_loss_rgb, correct_rgb = criterion(feat_rgb, labels_rgb, classifier)
            loss_ir, id_loss_ir, wrt_loss_ir, correct_ir = criterion(feat_ir, labels_ir, classifier)

            loss = loss_rgb + loss_ir
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_id_loss += (id_loss_rgb + id_loss_ir).item()
            total_wrt_loss += (wrt_loss_rgb + wrt_loss_ir).item()
            total_correct += (correct_rgb + correct_ir)
        return total_loss/len(loader), total_id_loss/len(loader), total_wrt_loss/len(loader), total_correct

    # -----------------------------
    # 特征提取与评估
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
        distmat = torch.cdist(feat_q, feat_g)
        dist_np = distmat.numpy()
        m = dist_np.shape[0]
        rank_counts = [0,0,0,0]
        INP_total = []
        for i in range(m):
            q_label = label_q[i].item()
            g_labels = label_g.numpy()
            sorted_idx = np.argsort(dist_np[i])
            matches = (g_labels[sorted_idx] == q_label).astype(int)

            for k, rank in enumerate([1,5,10,20]):
                if matches[:rank].sum() > 0:
                    rank_counts[k] += 1

            indices = np.where(matches==1)[0]
            if len(indices) > 0:
                INP_total.append(1.0 / (indices[-1]+1))
        mINP = np.mean(INP_total)
        return [c/m for c in rank_counts], mINP

    # -----------------------------
    # 主训练循环
    # -----------------------------
    num_epochs = 20  # 测试用
    for epoch in range(num_epochs):
        loss, id_loss, wrt_loss, correct = train_one_epoch(model, train_loader, optimizer, criterion, classifier)
        print(f"[Epoch {epoch+1}] 总Loss:{loss:.4f} IDLoss:{id_loss:.4f} WRTLoss:{wrt_loss:.4f} Correct:{correct}")

        # 提取特征并评估
        feat_q, label_q = extract_features(model, query_loader)
        feat_g, label_g = extract_features(model, gallery_loader)
        ranks, mINP_val = evaluate(feat_q, label_q, feat_g, label_g)
        print(f"Rank-1:{ranks[0]:.4f} Rank-5:{ranks[1]:.4f} Rank-10:{ranks[2]:.4f} Rank-20:{ranks[3]:.4f} mINP:{mINP_val:.4f}")
