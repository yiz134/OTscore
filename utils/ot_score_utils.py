import torch
import numpy as np
from tqdm import tqdm
from torch import linalg as LA

class OT_SCORE:
    def __init__(self, src_samples, src_labels, tar_samples, tar_labels, p=1):
        self.src_samples = src_samples
        self.src_labels = src_labels
        self.tar_samples = tar_samples
        self.tar_labels = tar_labels
        self.p = p
        self.num_classes = int(src_labels.max().item()) + 1
        self.reweight_factors = torch.zeros(len(self.src_samples), 1, device=self.src_samples.device)
        self.src_weights = torch.zeros_like(self.src_labels, dtype=torch.float32).to('cuda')
        self.reweight()

    def reweight(self):
        counts = torch.bincount(self.tar_labels, minlength=self.num_classes)
        total = counts.sum().float()
        class_proportions = counts.float() / total
        class_proportions = class_proportions.detach()
        
        for c in range(self.num_classes):
            class_mask = (self.src_labels == c)
            count = class_mask.sum().item()
            if count > 0:
                self.src_weights[class_mask] = class_proportions[c] / count
            else:
                self.src_weights[class_mask] = 0.0


    def compute_semi_discrete_OT(self, lr=None, max_iter=2000, batch_size=2000, epsilon=0.5):
        #self.reweight()
        self.tar_samples = self.tar_samples.to('cuda')
        if lr is None:
            lr_0 = 1.0
            l_0 = 100
            lr = lambda i: lr_0 / (1 + i / l_0)
        else:
            step_size = lr
            lr = lambda i: step_size
        for i in range(max_iter):
            tar_batch = self.tar_samples[torch.randperm(len(self.tar_samples))[:batch_size]]
            weighted_dist = (-torch.cdist(tar_batch, self.src_samples,
                                          p=2).T + self.reweight_factors)  # size = num of discrete points * B
            weighted_dist = weighted_dist / epsilon
            weighted_dist = weighted_dist - weighted_dist.max(dim=0, keepdim=True).values
            kai = torch.exp(weighted_dist)  # size = num of discrete points * B
            kai_normalized = kai / (kai.sum(dim=0, keepdim=True) + 1e-10)
            increment = lr(i) * (- torch.mean(kai_normalized, 1) + self.src_weights)
            self.reweight_factors.data = self.reweight_factors.data + increment.unsqueeze(1)
        self.reweight_factors = self.reweight_factors.data
        self.tar_samples = self.tar_samples.to('cpu')
        return self.reweight_factors
    
    def compute_ot_score(self, target_batch=None, target_batch_labels=None, format="indices", indices=None):
        self.tar_samples = self.tar_samples.to('cuda')
        if format=="indices":
            target_batch = self.tar_samples[indices]
            target_batch_labels = self.tar_labels[indices]
        ot_score = 100000+torch.zeros(len(target_batch), self.num_classes, device=self.src_samples.device)
        for cls_nu in range(self.num_classes):
            target_batch_cls, tar_mask = self.get_cls_samples(target_batch, target_batch_labels, cls_nu)
            if target_batch_cls.numel() == 0:
                continue
            src_samples_cls, mask = self.get_cls_samples(self.src_samples, self.src_labels, cls_nu)
            if src_samples_cls.numel() == 0:
                continue
            reweight_factors_cls = self.reweight_factors[mask]
            d_xy = torch.cdist(target_batch_cls, src_samples_cls) - reweight_factors_cls.T

            min_dxy, _ = torch.min(d_xy, dim=1)
            for cls_mu in range(self.num_classes):
                if cls_mu == cls_nu:
                    continue
                src_samples_cls, mask = self.get_cls_samples(self.src_samples, self.src_labels, cls_mu)
                if src_samples_cls.numel() == 0:
                    continue
                reweight_factors_cls = self.reweight_factors[mask]
                d_xz = torch.cdist(target_batch_cls, src_samples_cls) - reweight_factors_cls.T
                min_dxz, _ = torch.min(d_xz, dim=1)
                ot_score[tar_mask, cls_mu] = (min_dxz - min_dxy)#.unsqueeze(1)
        ot_score, _ = torch.min(ot_score, dim=1)
        self.tar_samples = self.tar_samples.to('cpu')
        return ot_score

    def get_cls_samples(self, samples, labels, cls):
        mask = (labels == cls)
        return samples[mask], mask

@torch.no_grad()
def compute_cls_mean_features(model, dataloader):
    model.eval()

    # Run inference
    features, gt_labels, pred_labels = [], [], []
    iterator = tqdm(dataloader)
    for _ , imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")

        feats, logits_cls = model(imgs, return_feats=True) #feats size B*D, logits_cls size B*C
        pred = logits_cls.argmax(dim=1) #size B
        features.append(feats)
        pred_labels.append(pred)
    
    cls_mean_features, cls_labels = compute_cls_mean_from_lists(features, pred_labels)
    return cls_mean_features, cls_labels


@torch.no_grad()
def compute_cls_mean_features_BFC(netB, netF, netC, dataloader, ratio=None):
    netB.eval()
    netF.eval()
    netC.eval()

    # Run inference
    features, gt_labels, pred_labels = [], [], []

    for imgs, labels, idxs in tqdm(dataloader):
        imgs = imgs.to("cuda") #
        feats = netB(netF(imgs))
        logits_cls = netC(feats)
        #feats, logits_cls = model(imgs, return_feats=True)  # feats size B*D, logits_cls size B*C
        pred = logits_cls.argmax(dim=1)  # size B
        features.append(feats)
        pred_labels.append(pred)
        gt_labels.append(labels.to("cuda"))
    if ratio is not None:
        print("Computing source features with ratio:", ratio)
        if not (0 < ratio <= 1):
            raise ValueError(f"ratio must be in (0, 1], but got {ratio}")
    
        if ratio < 1:
            all_features = torch.cat(features, dim=0)
            all_gt_labels = torch.cat(gt_labels, dim=0)
            all_pred_labels = torch.cat(pred_labels, dim=0)
    
            selected_indices = []
            for c in torch.unique(all_gt_labels):
                cls_idx = torch.nonzero(all_gt_labels == c, as_tuple=False).squeeze(1)
                num_keep = max(1, int(len(cls_idx) * ratio))
                perm = torch.randperm(len(cls_idx), device=cls_idx.device)
                selected_indices.append(cls_idx[perm[:num_keep]])
    
            selected_indices = torch.cat(selected_indices, dim=0)
            perm_all = torch.randperm(len(selected_indices), device=selected_indices.device)
            selected_indices = selected_indices[perm_all]
    
            features = [all_features[selected_indices]]
            gt_labels = [all_gt_labels[selected_indices]]
            pred_labels = [all_pred_labels[selected_indices]]
    # Source prototypes are aggregated by GT labels for better class coverage.
    cls_mean_features, cls_labels = compute_cls_mean_from_lists(features, gt_labels)
    return cls_mean_features, cls_labels


def compute_cls_mean_from_lists(features, pred_labels):
    features = torch.cat(features, dim=0)       # [N, D]
    pred_labels = torch.cat(pred_labels, dim=0) # [N]

    N, D = features.shape
    C = int(pred_labels.max().item()) + 1      

    sum_features = torch.zeros(C, D, device=features.device, dtype=features.dtype)
    counts = torch.zeros(C, device=features.device, dtype=torch.long)

    sum_features.index_add_(0, pred_labels, features)
    counts += torch.bincount(pred_labels, minlength=C)

    valid_mask = counts > 0
    cls_labels = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)  # [K]
    cls_mean_features = sum_features[cls_labels] / counts[cls_labels].unsqueeze(1)

    return cls_mean_features, cls_labels

@torch.no_grad()
def extract_features(model, dataloader):
    model.eval()

    # Run inference
    features, indices, pred_labels = [], [], []
    iterator = tqdm(dataloader)
    for _, imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")

        feats, logits_cls = model(imgs, return_feats=True)  # feats size B*D, logits_cls size B*C
        pred = logits_cls.argmax(dim=1)
        features.append(feats.cpu())
        indices.append(idxs)
        pred_labels.append(pred)

    features = torch.cat(features, dim=0)  # [N, D]
    indices = torch.cat(indices, dim=0)  # [N]
    pred_labels = torch.cat(pred_labels, dim=0)  # [N]
    sorted_idx = torch.argsort(indices)
    features = features[sorted_idx]
    pred_labels = pred_labels[sorted_idx]
    # indices = indices[sorted_idx]
    return features, pred_labels

@torch.no_grad()
def extract_features_BFC(netB, netF, netC, dataloader):
    netB.eval()
    netF.eval()
    netC.eval()

    # Run inference
    features, indices, pred_labels = [], [], []
    iterator = tqdm(dataloader)
    for imgs, labels, idxs in iterator:
        imgs = imgs.to("cuda")
        feats = netB(netF(imgs))
        logits_cls = netC(feats)
        #feats, logits_cls = model(imgs, return_feats=True)  # feats size B*D, logits_cls size B*C
        pred = logits_cls.argmax(dim=1)
        features.append(feats.cpu())
        indices.append(idxs)
        pred_labels.append(pred)

    features = torch.cat(features, dim=0)  # [N, D]
    indices = torch.cat(indices, dim=0)  # [N]
    pred_labels = torch.cat(pred_labels, dim=0)  # [N]
    sorted_idx = torch.argsort(indices)
    features = features[sorted_idx]
    pred_labels = pred_labels[sorted_idx]
    # indices = indices[sorted_idx]
    return features, pred_labels

import matplotlib.pyplot as plt
from pathlib import Path
def plot_conf_score(score, args, normalize=False):
    score_cpu = score.to('cpu')
    if normalize:
        min_score = score_cpu.min()
        max_score = score_cpu.max()
        score_cpu = (score_cpu - min_score) / (max_score - min_score)

    # plot
    fig, ax = plt.subplots()
    ax.hist(score_cpu.numpy(), bins=50)
    ax.set_xlabel("Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Histogram: mean={:.2f}".format(score_cpu.mean().item()))
    fig.tight_layout()
    save_name = f"{args.coeff}_{args.names[args.s]}2{args.names[args.t]}_epoch{int(args.current_epoch)}.png"
    pair_dir = f"{args.names[args.s]}2{args.names[args.t]}"
    save_path = Path("plots") / str(args.coeff) / pair_dir / save_name
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=200, bbox_inches="tight")