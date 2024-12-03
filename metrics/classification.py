import numpy as np



def confusion_matrix(y_true, y_pred, positive_label=None):

    """
    Computes the confusion matrix for binary classification.

    Parameters:
    y_true (numpy.ndarray): Array of true labels.
    y_pred (numpy.ndarray): Array of predicted labels (discrete values).
    positive_label (int or str): The label representing the positive class.

    Returns:
    numpy.ndarray: Confusion matrix as a 2x2 array:
                    [[TP, FN],
                     [FP, TN]]
    """

    assert np.issubdtype(y_pred.dtype, np.integer), "Predictions should be discrete labels, not continuous probabilities."
    if positive_label is None:
        if len(np.unique(y_true)) == 2:
            positive_label = 1
        else:
            raise Exception("For multiclass, please provide the positive label.")
    else:
         assert positive_label in y_true, "Chosen positive label is not valid."

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(y_true)):
        if y_pred[i] == positive_label:
            if y_true[i] == positive_label:
                tp += 1
            else:
                fp += 1
        else:
            if y_true[i] != positive_label:
                tn += 1
            else:
                fn += 1
    return np.array([[tp, fn], [fp, tn]])





def accuracy_score(y_true, y_pred):

    """

    Computes the accuracy score for binary or multi-class classification.

    In binary classification, accuracy is calculated as:
        (TP + TN) / (TP + TN + FP + FN)
    For multi-class classification, accuracy is computed for each class 
    and then averaged across all classes.

    Parameters:
    y_true (numpy.ndarray): Array of true labels.
    y_pred (numpy.ndarray): Array of predicted labels.

    Returns:
    float: Accuracy score, either as a single value for binary classification 
           or the average for multi-class classification.

    """

    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:
        confusion = confusion_matrix(y_true, y_pred, positive_label = 1)
        tp, fn, fp, tn = confusion.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
    else:
        accuracies = []
        for label in unique_labels:
            confusion = confusion_matrix(y_true, y_pred, positive_label = label)
            tp, fn, fp, tn = confusion.ravel()
            label_accuracy = (tp + tn) / (tp + tn + fp + fn + 1e-8)
            accuracies.append(label_accuracy)
        accuracy = np.mean(accuracies)
    return accuracy
            



def precision_score(y_true, y_pred):

    """

    Calculate the precision score for binary or multiclass classification.
    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of the data.
    y_pred : array-like of shape (n_samples,)
        Predicted labels of the data.
    Returns:
    --------
    precision : float
        Precision score, which is the ratio of true positives to the sum of true positives and false positives.
        For binary classification, it returns the precision for the positive class.
        For multiclass classification, it returns the average precision across all classes.

    """

    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:
        confusion = confusion_matrix(y_true, y_pred, positive_label = 1)
        tp, fn, fp, tn = confusion.ravel()
        precision = tp / (tp + fp + 1e-8)
    else:
        precisions = []
        for label in unique_labels:
            confusion = confusion_matrix(y_true, y_pred, positive_label = label)
            tp, fn, fp, tn = confusion.ravel()
            label_precision = tp / (tp + fp + 1e-8)
            precisions.append(label_precision)
        precision = np.mean(precisions)
    return precision




def recall_score(y_true, y_pred):

    """

    Calculate the recall score for binary or multiclass classification.

    Parameters:
    -----------
    y_true : array-like of shape (n_samples,)
        True labels of the data.
    y_pred : array-like of shape (n_samples,)
        Predicted labels of the data.

    Returns:
    --------
    recall : float
        Recall score, which is the ratio of true positives to the sum of true positives and false negatives.
        For binary classification, it returns the recall for the positive class.
        For multiclass classification, it returns the average recall across all classes.

    """

    unique_labels = np.unique(y_true)

    if len(unique_labels) == 2:
        confusion = confusion_matrix(y_true, y_pred, positive_label = 1)
        tp, fn, fp, tn = confusion.ravel()
        recall = tp / (tp + fn + 1e-8)
    else:
        recalls = []
        for label in unique_labels:
            confusion = confusion_matrix(y_true, y_pred, positive_label = label)
            tp, fn, fp, tn = confusion.ravel()
            label_recall = tp / (tp + fn + 1e-8)
            recalls.append(label_recall)
        recall = np.mean(recalls)
    return recall





def f1_score(y_true, y_pred):

    """

    Calculate the F1 score, which is the harmonic mean of precision and recall.
    Parameters:
    y_true (list or array-like): True labels.
    y_pred (list or array-like): Predicted labels.
    Returns:
    float: F1 score.

    """

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 / (1 / precision + 1 / recall + 1e-8)





def classification_score(y_true, y_pred):

    """

    Calculate various classification metrics for given true and predicted labels.
    Parameters:
    y_true (array-like): True labels.
    y_pred (array-like): Predicted labels.
    Returns:
    dict: A dictionary containing the following classification metrics:
        - "accuracy": Accuracy score.
        - "precision": Precision score.
        - "recall": Recall score.
        - "f1": F1 score.

    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred)
    }


def log_loss(y_true, y_pred):
    """
    Compute the logistic loss (log loss) between true labels and predicted probabilities.

    Log loss, also known as logistic regression loss or cross-entropy loss, measures the performance of a classification model
    where the prediction is a probability value between 0 and 1. The loss increases as the predicted probability diverges from
    the actual label.

    Parameters:
    y_true (array-like): True binary labels in range {0, 1}.
    y_pred (array-like): Predicted probabilities, as returned by a classifier's predict_proba method.

    Returns:
    float: The computed log loss value.
    """
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1-epsilon)
    return - np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

