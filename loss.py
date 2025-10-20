import torch
import torch.nn as nn

# --------------------------
# 辅助函数
# --------------------------
def normalize(x, axis=-1):
    """将特征向量归一化到单位长度"""
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def pdist_torch(emb1, emb2):
    """
    计算欧氏距离矩阵
    emb1: (m, C)
    emb2: (n, C)
    返回: (m, n)
    """
    m, n = emb1.shape[0], emb2.shape[0]
    emb1_pow = torch.pow(emb1, 2).sum(dim=1, keepdim=True).expand(m, n)
    emb2_pow = torch.pow(emb2, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    dist_mtx = emb1_pow + emb2_pow
    dist_mtx = dist_mtx.addmm(emb1, emb2.t(), beta=1.0, alpha=-2.0)
    dist_mtx = dist_mtx.clamp(min=1e-12).sqrt()
    return dist_mtx

def softmax_weights(dist, mask):
    """
    计算 WRT 的权重
    dist: 距离矩阵
    mask: 0/1 mask，选择正样本或负样本
    """
    max_v = torch.max(dist * mask, dim=1, keepdim=True)[0]
    diff = dist - max_v
    Z = torch.sum(torch.exp(diff) * mask, dim=1, keepdim=True) + 1e-6
    W = torch.exp(diff) * mask / Z
    return W

# --------------------------
# Weighted Regularized Triplet Loss
# --------------------------
class TripletLoss_WRT(nn.Module):
    """Weighted Regularized Triplet Loss"""

    def __init__(self):
        super(TripletLoss_WRT, self).__init__()
        self.ranking_loss = nn.SoftMarginLoss()

    def forward(self, inputs, targets, normalize_feature=True):
        if normalize_feature:
            inputs = normalize(inputs, axis=-1)
        dist_mat = pdist_torch(inputs, inputs)

        N = dist_mat.size(0)
        is_pos = targets.expand(N, N).eq(targets.expand(N, N).t()).float()
        is_neg = targets.expand(N, N).ne(targets.expand(N, N).t()).float()

        dist_ap = dist_mat * is_pos
        dist_an = dist_mat * is_neg

        weights_ap = softmax_weights(dist_ap, is_pos)
        weights_an = softmax_weights(-dist_an, is_neg)
        furthest_positive = torch.sum(dist_ap * weights_ap, dim=1)
        closest_negative = torch.sum(dist_an * weights_an, dim=1)

        y = furthest_positive.new().resize_as_(furthest_positive).fill_(1)
        loss = self.ranking_loss(closest_negative - furthest_positive, y)

        correct = torch.ge(closest_negative, furthest_positive).sum().item()
        return loss, correct

# --------------------------
# 综合 AGW 损失：ID Loss + WRT Loss
# --------------------------
class AGW_Loss(nn.Module):
    """
    综合 ID Loss 和 WRT Loss 的损失函数
    """

    def __init__(self, num_classes, lambda_wrt=1.0):
        super(AGW_Loss, self).__init__()
        self.lambda_wrt = lambda_wrt
        self.id_criterion = nn.CrossEntropyLoss()
        self.wrt_criterion = TripletLoss_WRT()
        self.num_classes = num_classes

    def forward(self, features, labels, classifier):
        """
        Args:
            features: AGW 输出特征, shape: (B, C)
            labels: 样本标签, shape: (B,)
            classifier: 分类头 nn.Linear(C, num_classes)
        Returns:
            total_loss: ID + WRT
            id_loss: 分类损失
            wrt_loss: 三元组损失
        """
        # ID Loss
        logits = classifier(features)
        id_loss = self.id_criterion(logits, labels)

        # WRT Loss
        wrt_loss, correct = self.wrt_criterion(features, labels, normalize_feature=True)

        # 总损失
        total_loss = id_loss + self.lambda_wrt * wrt_loss
        return total_loss, id_loss, wrt_loss, correct
