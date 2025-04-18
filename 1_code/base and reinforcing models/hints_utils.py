from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
import pandas as pd

def get_input_shape(input_shape_str):
    str_list = input_shape_str.split(',')
    w = int(str_list[0])
    h = int(str_list[1])
    d = int(str_list[2])
    return w,h,d

def eopp(recall_A, recall_B, specificity_A, specificity_B):
    eopp0 = abs(recall_A - recall_B)
    eopp1 = abs(specificity_A - specificity_B)
    eodd = eopp0 + eopp1
    return eopp0, eopp1, eodd

def precision(fp, tp):
    precision = tp / (tp + fp)
    return precision

def specificity(tn, fp):
    specificity = tn / (tn + fp)
    return specificity

def recall(fn, tp):
    recall = tp / (tp + fn)
    return recall

def f1_score(precision, recall):
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score

def get_subgroup_metrics(prediction_file, df_gt, sex):
    gender_data = df_gt[df_gt['sex'] == sex]
    df2 = pd.read_csv(prediction_file)
    merged_data = gender_data.merge(df2, on='isic_id', how='inner')
    auc = roc_auc_score(merged_data['true_label'],merged_data['prediction'])
    merged_data['prediction'] = merged_data['prediction'].round().astype(int)
    tn, fp, fn, tp = confusion_matrix(merged_data['true_label'], merged_data['prediction']).ravel()
    return auc, tn, fp, fn, tp

def report(tn, fp, fn, tp, auc, sex):
    print (sex + ', auc = ' + str(round(auc,3)) + ' , TN = ' + str(tn) + ' , FP = ' + str(fp) + ' , FN = ' + str(fn) + ' , TP = ' + str(tp))    
    prec = precision(fp,tp)
    spec = specificity(tn, fp)
    reca = recall(fn, tp)
    f1 = f1_score(prec, reca)
    print('Precision = ' + str(round(prec,3)))
    print('Specificity = ' + str(round(spec,3)))
    print('Recall = ' + str(round(reca,3)))
    print('F1 score = ' + str(round(f1,3)))
