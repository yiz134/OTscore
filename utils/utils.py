from pathlib import Path
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import auc

def write_aurc_to_excel(args, scores: dict):
    score_order = ["Maxprob", "Ent", "Cossim", "JMDS", "OT Score"]
    vals = [scores.get(k, None) for k in score_order]

    row_label = f"{int(args.s)}_to_{int(args.t)}"
    columns = ["s_to_t"] + score_order
    row = [row_label] + vals

    xlsx_path = Path(f"{args.dset}.xlsx")
    sheet_name = str(args.seed)

    if xlsx_path.exists():
        try:
            df_old = pd.read_excel(xlsx_path, sheet_name=sheet_name)
            for col in columns:
                if col not in df_old.columns:
                    df_old[col] = pd.NA
            df_old = df_old[columns]
            df_new = pd.concat([df_old, pd.DataFrame([row], columns=columns)],
                               ignore_index=True)
        except ValueError:
            df_new = pd.DataFrame([row], columns=columns)

        with pd.ExcelWriter(xlsx_path, engine="openpyxl",
                            mode="a", if_sheet_exists="replace") as writer:
            df_new.to_excel(writer, sheet_name=sheet_name, index=False)
    else:
        df = pd.DataFrame([row], columns=columns)
        with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)


def Maxprob(prob_X):
    """
    Get the maximum class probability for each sample in a PyTorch tensor.

    Parameters:
    prob_X (torch.Tensor): A tensor of shape (n, c) where:
        - n is the number of samples
        - c is the number of classes
        - Each row represents a probability distribution over `c` classes

    Returns:
    torch.Tensor: A tensor of shape (n,) containing the maximum probability for each sample.
    """
    if not isinstance(prob_X, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    if prob_X.dim() != 2:
        raise ValueError("Input tensor must be two-dimensional (n, c)")

    return torch.max(prob_X, dim=1).values  # Get max probability for each sample


def Ent(prob_X):
    if not isinstance(prob_X, torch.Tensor):
        raise ValueError("Input must be a PyTorch tensor")

    if prob_X.dim() != 2:
        raise ValueError("Input tensor must be two-dimensional (n, c)")

    prob_X = prob_X / prob_X.sum(dim=1, keepdim=True)

    K = prob_X.shape[1]

    epsilon = 1e-5
    log_prob_X = torch.log(prob_X + epsilon)  # log of probability

    entropy = torch.sum(prob_X * log_prob_X, dim=1)

    normalized_entropy = 1 + entropy / torch.log(torch.tensor(K, dtype=torch.float32))

    return normalized_entropy

def compute_cossim_score(x_t, num_classes, pseudo_labels=None):
    device = x_t.device
    x_norm = F.normalize(x_t, dim=1)

    generated_pseudo = False

    if pseudo_labels is None:
        x_np = x_norm.detach().cpu().numpy()
        kmeans = KMeans(
            n_clusters=num_classes,
            n_init=10,
            random_state=0,
            max_iter=300
        )
        pseudo_labels_np = kmeans.fit_predict(x_np)
        pseudo_labels = torch.tensor(pseudo_labels_np, dtype=torch.long, device=device)

        generated_pseudo = True

        centers = torch.zeros((num_classes, x_t.size(1)), device=device)
        for c in range(num_classes):
            mask = (pseudo_labels == c)
            if mask.sum() > 0:
                centers[c] = x_t[mask].mean(dim=0)
    else:
        centers = torch.zeros((num_classes, x_t.size(1)), device=device)
        for c in range(num_classes):
            mask = (pseudo_labels == c)
            if mask.sum() > 0:
                centers[c] = x_t[mask].mean(dim=0)

    centers = F.normalize(centers, dim=1)
    selected_centers = centers[pseudo_labels]
    cos_sim = (x_norm * selected_centers).sum(dim=1)
    cossim_score = 0.5 * (1 + cos_sim)

    if generated_pseudo:
        return cossim_score, pseudo_labels
    else:
        return cossim_score


def compute_scores(net_prob=None, gmm_label=None, feat=None, class_num=None):
    score_dict={}
    score_dict["Maxprob"] = Maxprob(net_prob)
    score_dict["Ent"] = Ent(net_prob)
    score_dict["Cossim"] = compute_cossim_score(feat, class_num, pseudo_labels=gmm_label)
    return score_dict

def compute_aurc(score_dict, net_pred=None, gmm_label=None, true_label=None, thresholds=None):
    all_aurc = dict.fromkeys(score_dict, None)
    for score_name, score in score_dict.items():
        if score_name in ["Maxprob", "Ent"]:
            pred_labels = net_pred
        else:
            pred_labels = gmm_label
        score = torch.as_tensor(score).clone().detach().cpu()
        pred_labels = torch.as_tensor(pred_labels).clone().detach().cpu()
        sorted_scores, sorted_indices = torch.sort(score, descending=True)
        sorted_y_true = true_label[sorted_indices]
        sorted_y_pred = pred_labels[sorted_indices]

        num_samples = len(score)
        risks, coverages = [], []

        for p in thresholds:
            keep_num = int((p / 100) * num_samples)
            if keep_num == 0:
                risks.append(0.0)
                coverages.append(0.0)
                continue

            kept_y_true = sorted_y_true[:keep_num]
            kept_y_pred = sorted_y_pred[:keep_num]
            incorrect = (kept_y_true.cpu() != kept_y_pred).float()
            risk = incorrect.mean().item()
            coverage = keep_num / num_samples

            risks.append(risk)
            coverages.append(coverage)

        # aurc = sum([(coverages[1] - coverages[0]) * r for r in risks[:-1]])#auc(coverages, risks)
        aurc = auc(coverages, risks)
        all_aurc[score_name] = aurc
    return all_aurc
