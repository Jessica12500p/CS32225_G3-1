import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from src.config import POSES, SEED


def candidates():
    return {
        'SVM-RBF': SVC(C=15, gamma='scale', cache_size=512, class_weight='balanced'),
        'ExtraTrees': ExtraTreesClassifier(n_estimators=180, max_features=.6,
                                          min_samples_leaf=2, n_jobs=4, random_state=SEED),
    }


def evaluate(model, x, y):
    pred = model.predict(x)
    report = classification_report(y, pred, labels=list(range(4)), target_names=POSES,
                                   output_dict=True, zero_division=0)
    return {'accuracy': float(accuracy_score(y,pred)), 'precision': report['macro avg']['precision'],
            'recall': report['macro avg']['recall'], 'f1': report['macro avg']['f1-score'],
            'per_class': report, 'confusion_matrix': confusion_matrix(y,pred,labels=range(4)).tolist(),
            'samples': len(y), 'target': .95, 'passed': bool(accuracy_score(y,pred) > .95)}
