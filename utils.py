import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, roc_auc_score, auc, precision_recall_curve, roc_curve, average_precision_score

## Define functions
def merge_feature(list_features):
    """
      Merge numpy array features
      Args:
        - list_features: list of numpy array features
                         :type: a list of numpy arrays 
      Returns:
        - features: the concatenate numpy array along axis=1
                    :type: a numpy array                 
    """      
    features = np.concatenate(list_features, axis=1)
    features = np.nan_to_num(features)
    features = np.clip(features, -np.finfo(np.float32).max, np.finfo(np.float32).max)
    return features

def compute_metrics(cfs_matrix):
    """
      Calculate common metrics based on the confusion matrix
      Args:
        - cfs_matrix: a sklearn confusion matrix 
                      :type: a ndarray of shape (n_classes, n_classes)
      Returns:
        - precision: the precision of the prediction
                     :type: float  
        - recall: the recall of the prediction
                  :type: float  
        - f1: the f1-score of the prediction
              :type: float                       
    """     
    precision = cfs_matrix[1,1] / (cfs_matrix[1,1] + cfs_matrix[0,1])
    recall = cfs_matrix[1,1] / (cfs_matrix[1,1] + cfs_matrix[1,0])
    f1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, f1

# # Evaluate
def evaluate(ensem_preds, targets):
    """
      Evaluate the prediction by providing metrics & also the best threshold (to get the highest f1-score)
      Ex: AUC, Accurary, Precision, Recall, F1-Score.
      Then print these metrics
      Args:
        - ensem_preds: predictions for ids 
                       :type: a numpy array
        - targets: the actual results of ids 
                   :type: a numpy array                 
      Returns:
        - None                  
    """     
    best_th = 0
    best_score = 0

    for th in np.arange(0.0, 0.6, 0.01):
        pred = (ensem_preds > th).astype(int)
        score = f1_score(targets, pred)
        if score > best_score:
            best_th = th
            best_score = score

    print(f"\nAUC score: {roc_auc_score(targets, ensem_preds):12.4f}")
    print(f"Best threshold {best_th:12.4f}")

    preds = (ensem_preds > best_th).astype(int)

    cm1 = confusion_matrix(targets, preds)
    print('\nConfusion Matrix : \n', cm1)
    precision, recall, f1 = compute_metrics(cm1)
    
    print('\n=============')
    print (f'Precision    : {precision:12.4f}')
    
    print(f'Recall : {recall:12.4f}')
    
    print(f'F1 Score : {f1:12.4f}')
    
    total1=sum(sum(cm1))

    print('\n=============')
    accuracy1=(cm1[0,0]+cm1[1,1])/total1
    print (f'Accuracy    : {accuracy1:12.4f}')