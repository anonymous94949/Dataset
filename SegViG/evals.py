import torch
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    cohen_kappa_score, classification_report,
)


def spearman_rank_correlation(y_true, y_pred):
    rank_true = y_true.argsort().argsort().float()
    rank_pred = y_pred.argsort().argsort().float()
    return torch.corrcoef(torch.stack((rank_true, rank_pred)))[0, 1]


def overall_result(true, pred):
    true = true.cpu()
    pred = pred.cpu()
    mae = torch.mean(torch.abs(true.float() - pred.float()))
    mse = torch.mean((true.float() - pred.float()) ** 2)
    kohen_quad = cohen_kappa_score(true, pred, weights='quadratic')
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro')
    classif_report = classification_report(true, pred)
    confusion_mat = confusion_matrix(true.tolist(), pred.tolist())
    spearman = spearman_rank_correlation(true, pred)
    return mae, mse, kohen_quad, spearman, acc, f1, classif_report, confusion_mat
