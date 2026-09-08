"""Reproducible end-to-end training: python -m src.train."""
import argparse
import json
import logging
import time
import joblib
import numpy as np
from src.config import LOGS, RESULTS, MODELS, SEED, ensure_dirs
from src.alg.data import load_raw, load_regions, split_groups, augment
from src.alg.features import features
from src.alg import posture, regions
from src.alg.identity import IdentityMatcher


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
    tmp.replace(path)


def save_model(name, model):
    path=MODELS/name
    tmp=path.with_suffix('.tmp')
    joblib.dump(model,tmp,compress=3)
    tmp.replace(path)


def run():
    ensure_dirs()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.FileHandler(LOGS/'training.log',encoding='utf-8'), logging.StreamHandler()])
    started = time.time()
    raw = load_raw()
    logging.info('读取 %s 帧 / %s 用户',len(raw.pressure),len(np.unique(raw.people)))
    train, test = split_groups(raw.people)
    train_people = sorted(np.unique(raw.people[train]).tolist())
    test_people = sorted(np.unique(raw.people[test]).tolist())
    report = {'seed':SEED,'shape':[44,24], 'raw_frames':len(raw.pressure),
              'train_users':train_people,'test_users':test_people,
              'augmentation':'仅训练帧增加一次 0.9–1.1 倍增益与 σ=1.5 噪声；已有镜像不重复增加',
              'pressure_unit':'原始传感器读数（未提供物理单位标定）',
              'limitations':['单次采集中的相邻帧存在相关性，区域同用户验证和用户识别不代表跨夜泛化。',
                              '区域准确率按 IoU≥0.5 定义，原任务未指定度量，另提供 mean IoU 与逐区域结果。']}
    logging.info('提取压力特征')
    x = features(raw.pressure)
    augmented_raw = augment(raw.pressure[train])
    np.savez_compressed(RESULTS/'augmented_posture_train.npz', pressure=augmented_raw,
                        poses=raw.poses[train], people=raw.people[train], parent_indices=train)
    xa = features(augmented_raw)
    inner_a, inner_b = split_groups(raw.people[train], fraction=.25)
    scores = {}
    for name, model in posture.candidates().items():
        logging.info('睡姿 %s: 训练用户内部选择',name)
        model.fit(np.concatenate([x[train[inner_a]],xa[inner_a]]), np.tile(raw.poses[train[inner_a]],2))
        scores[name] = posture.evaluate(model,x[train[inner_b]],raw.poses[train[inner_b]])['accuracy']
    best = max(scores,key=scores.get)
    report['posture'] = {'selection':'训练用户内部按用户划分验证，最终测试用户不参与模型选择',
                         'validation_scores':scores,'best_model':best,'models':{}}
    for name, model in posture.candidates().items():
        logging.info('睡姿 %s: 全训练用户拟合',name)
        model.fit(np.concatenate([x[train],xa]), np.tile(raw.poses[train],2))
        report['posture']['models'][name] = posture.evaluate(model,x[test],raw.poses[test])
        save_model(f'posture_{name}.joblib',model)
        logging.info('%s 测试 accuracy=%.4f',name,report['posture']['models'][name]['accuracy'])
    write_json(RESULTS/'training_progress.json',report)
    logging.info('训练五区域回归')
    region = load_regions()
    known = np.flatnonzero(np.isin(region.people,train_people))
    new = np.flatnonzero(np.isin(region.people,test_people))
    ri, rv = split_groups(region.groups[known])
    rt, rv = known[ri],known[rv]
    rx = features(region.pressure)
    rm = regions.make_model()
    augmented_regions = augment(region.pressure[rt])
    np.savez_compressed(RESULTS/'augmented_region_train.npz',pressure=augmented_regions,
                        boxes=region.boxes[rt], parent_indices=rt)
    rm.fit(np.concatenate([rx[rt],features(augmented_regions)]),
           np.tile(region.boxes[rt].reshape(-1,20),(2,1)))
    report['regions'] = {'definition':'矩形边界按零起点、右下边界不包含；五区域不含小腿',
                         'split':'先隔离新用户；已知用户按原帧及镜像家族分组，70%训练、30%验证',
                         'train_samples':len(rt), 'validation':regions.evaluate(rm,rx[rv],region.boxes[rv],.95),
                         'new_users':regions.evaluate(rm,rx[new],region.boxes[new],.7)}
    save_model('regions.joblib',rm)
    write_json(RESULTS/'training_progress.json',report)
    logging.info('区域验证 %.4f / 新用户 %.4f',report['regions']['validation']['accuracy'],report['regions']['new_users']['accuracy'])
    logging.info('训练用户模板匹配及独立阈值校准')
    ii, ih = split_groups(raw.groups[train],fraction=.4)
    ic, it = split_groups(raw.groups[train[ih]],fraction=.5,seed=SEED+1)
    enroll, calibrate, identity_test = train[ii], train[ih[ic]], train[ih[it]]
    unknown_cal_people = test_people[:len(test_people)//2]
    unknown_test_people = test_people[len(test_people)//2:]
    uc = np.flatnonzero(np.isin(raw.people,unknown_cal_people))
    ut = np.flatnonzero(np.isin(raw.people,unknown_test_people))
    matcher = IdentityMatcher().fit(np.concatenate([x[enroll],features(augment(raw.pressure[enroll]))]), np.tile(raw.people[enroll],2))
    calibration = matcher.calibrate(x[calibrate],raw.people[calibrate],x[uc])
    report['identity'] = matcher.evaluate(x[identity_test],raw.people[identity_test],x[ut])
    report['identity'].update({'calibration':calibration,'enrolled_users':train_people,
                              'unknown_calibration_users':unknown_cal_people,'unknown_test_users':unknown_test_people,
                              'split':'已知用户原帧及镜像家族按约 60/20/20 注册/校准/测试；未知用户校准与测试隔离',
                              'enrollment_samples':len(enroll)})
    save_model('identity.joblib',matcher)
    sample = identity_test[::max(1,len(identity_test)//400)]
    unknown_sample = ut[::max(1,len(ut)//100)]
    sample = np.concatenate([sample,unknown_sample])
    xy = matcher.transform(x[sample])[:,:2]
    predicted, distances = matcher.predict(x[sample])
    write_json(RESULTS/'identity_projection.json',{'axes':'PCA 1 / PCA 2（仅注册数据拟合）', 'points':[
        {'x':float(pt[0]),'y':float(pt[1]),'person':str(raw.people[i]),'known':bool(raw.people[i] in train_people),
         'prediction':str(predicted[j]),'distance':float(distances[j])} for j,(i,pt) in enumerate(zip(sample,xy))]})
    report['duration_seconds'] = round(time.time()-started,2)
    report['completed'] = True
    write_json(RESULTS/'training_progress.json',report)
    write_json(RESULTS/'metrics.json',report)
    write_json(RESULTS/'split_manifest.json',{'seed':SEED,'train_users':train_people,'test_users':test_people,
        'posture_train_indices':train.tolist(),'posture_test_indices':test.tolist(),
        'region_train_indices':rt.tolist(),'region_validation_indices':rv.tolist(),'region_new_user_indices':new.tolist(),
        'identity_enrollment_indices':enroll.tolist(),'identity_calibration_indices':calibrate.tolist(),
        'identity_test_indices':identity_test.tolist(),'unknown_calibration_indices':uc.tolist(),'unknown_test_indices':ut.tolist()})
    lines = ['# 实测算法报告','',f'随机种子 {SEED}；训练用户 {len(train_people)}，测试用户 {len(test_people)}。',
             '', '| 任务 | 实测 | 目标 | 达标 |','|---|---:|---:|---|']
    for name,result in report['posture']['models'].items():
        lines.append(f'| 睡姿 {name} | {result["accuracy"]:.2%} | >95% | {result["passed"]} |')
    for key,label in [('validation','区域同用户验证'),('new_users','区域新用户')]:
        result=report['regions'][key]
        lines.append(f'| {label}（IoU≥0.5） | {result["accuracy"]:.2%} | >{result["target"]:.0%} | {result["passed"]} |')
    ident=report['identity']
    lines.extend([f'| 用户闭集识别 | {ident["accuracy"]:.2%} | >95% | {ident["accuracy"]>.95} |',
                  f'| FAR | {ident["far"]:.2%} | 1–5% | { .01<=ident["far"]<=.05} |',
                  f'| FRR | {ident["frr"]:.2%} | 5–10% | {.05<=ident["frr"]<=.10} |','',
                  '## 评估说明','',report['augmentation'],'',report['posture']['selection'],'',report['regions']['split'],
                  '',report['identity']['split'],'','\n'.join(report['limitations']),
                  '', '未达标项不能视为验收通过；下一步应增加跨用户训练样本、采用可学习空间特征，并以独立跨夜数据验证身份阈值。',
                  '', '完整精确率、召回率、F1、混淆矩阵与划分索引见 metrics.json / split_manifest.json。'])
    (RESULTS/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    logging.info('训练结束，用时 %.1fs，结果位于 %s',time.time()-started,RESULTS)


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
