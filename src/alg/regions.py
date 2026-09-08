import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from src.config import REGIONS, SEED


def make_model():
    return ExtraTreesRegressor(n_estimators=120, max_features=.8, min_samples_leaf=2,
                              max_depth=28, n_jobs=4, random_state=SEED)


def predict_boxes(model, x):
    boxes = model.predict(x).reshape(-1,5,4)
    boxes = np.clip(boxes, 0, [24,44,24,44])
    lo = np.minimum(boxes[:,:,:2], boxes[:,:,2:])
    hi = np.maximum(boxes[:,:,:2], boxes[:,:,2:])
    return np.concatenate([lo,hi],axis=2)


def iou(truth, pred):
    intersection = np.maximum(np.minimum(truth[:,:,2:],pred[:,:,2:])-np.maximum(truth[:,:,:2],pred[:,:,:2]),0).prod(axis=2)
    union = (truth[:,:,2:]-truth[:,:,:2]).prod(axis=2)+(pred[:,:,2:]-pred[:,:,:2]).prod(axis=2)-intersection
    return intersection / np.maximum(union,1e-8)


def evaluate(model, x, boxes, target):
    pred = predict_boxes(model,x)
    scores = iou(boxes,pred)
    success = float((scores >= .5).mean())
    return {'accuracy': success, 'metric': '五区域矩形 IoU≥0.5 的比例（不含背景与小腿）',
            'mean_iou': float(scores.mean()), 'all_five_success': float((scores>=.5).all(axis=1).mean()),
            'mae_cells': float(np.abs(boxes-pred).mean()), 'target': target, 'passed': success>target,
            'samples': len(x), 'per_region': {name:{'mean_iou':float(scores[:,i].mean()),
            'accuracy':float((scores[:,i]>=.5).mean())} for i,name in enumerate(REGIONS)}}
