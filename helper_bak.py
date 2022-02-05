'''
'''

import numpy as np
from sklearn import metrics

def train_roc_pr(y_pred_cv):
    
    from sklearn import metrics
    
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 500)
    
    precisions = []
    aps = []
    mean_recalls = np.linspace(0, 1, 500) 
    for i_y_train, i_train_pred in y_pred_cv:
        i_fpr, i_tpr, _ = metrics.roc_curve(i_y_train, i_train_pred) #drop_intermediate=False)
        i_auc = metrics.auc(i_fpr, i_tpr)
        
        interp_tpr = np.interp(mean_fpr, i_fpr, i_tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(i_auc)
        
        i_precision, i_recall, _ = metrics.precision_recall_curve(i_y_train, i_train_pred)
        i_average_precision = metrics.average_precision_score(i_y_train, i_train_pred)
        
        interp_precision = np.interp(mean_recalls, i_recall, i_precision,period=np.inf)
        interp_precision[0] = 1.0
        precisions.append(interp_precision)
        aps.append(i_average_precision)
    
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    std_tpr = np.std(tprs, axis=0)
    
    mean_auc = np.mean(aucs)#auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    
    mean_precision = np.mean(precisions, axis=0)
#     mean_precision[-1] = 0.0
    std_precision = np.std(precisions, axis=0)
    
    mean_ap = np.mean(aps)
    std_ap = np.std(aps)
    
    return mean_tpr,std_tpr,mean_fpr,mean_auc,std_auc, mean_precision, std_precision, mean_recalls, mean_ap, std_ap