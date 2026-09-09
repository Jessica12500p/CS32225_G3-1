import json
from functools import lru_cache
from threading import RLock
import joblib
import numpy as np
from src.config import DATA, MODELS, RESULTS, POSES, REGIONS
from src.alg.data import load_raw, load_regions, parse_txt
from src.alg.features import features
from src.alg.regions import predict_boxes
from src.alg.telemetry import pressure_metrics, motion_state, simulate_airbags


class MattressService:
    def __init__(self):
        self.raw=load_raw()
        self.lock=RLock()
        self.dynamic_paths=sorted((DATA/'Spos_data').glob('*/*动态*.txt'))

    def report(self):
        path=RESULTS/'metrics.json'
        return json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'completed':False}

    @lru_cache(maxsize=6)
    def model(self,name,mtime):
        return joblib.load(MODELS/name)

    def get_model(self,name):
        path=MODELS/name
        return self.model(name,path.stat().st_mtime_ns) if path.exists() else None

    @lru_cache(maxsize=1)
    def annotations(self):
        return load_regions()

    @lru_cache(maxsize=4)
    def dynamic(self,index):
        if not 0<=index<len(self.dynamic_paths):
            raise ValueError('动态文件不存在')
        return parse_txt(self.dynamic_paths[index])

    def sessions(self):
        return [{'id':i,'name':p.stem,'person':p.parent.name,'frames':len(self.dynamic(i)),
                 'duration':70,'timing':'按文档 70 秒等间隔回放，原文件无逐帧时间戳'} for i,p in enumerate(self.dynamic_paths)]

    def select(self,person,action,frame):
        indices=np.flatnonzero((self.raw.people==person)&(self.raw.actions==action))
        if not len(indices) or not 0<=frame<len(indices):
            raise ValueError('用户、动作或帧号不存在')
        return indices,indices[frame]

    def infer(self,p,previous=None,elapsed=0.,strength=.5):
        x=features(p)
        report=self.report()
        best=report.get('posture',{}).get('best_model')
        pm=self.get_model(f'posture_{best}.joblib') if best else None
        pose=int(pm.predict(x)[0]) if pm else None
        previous_pose=int(pm.predict(features(previous))[0]) if pm is not None and previous is not None else pose
        state,change=motion_state(p,previous,pose,previous_pose)
        rm=self.get_model('regions.joblib')
        predicted=predict_boxes(rm,x)[0].round(2).tolist() if rm is not None else None
        im=self.get_model('identity_enrolled.joblib') or self.get_model('identity.joblib')
        identity=None
        if im is not None:
            with self.lock:
                names,distances=im.predict(x)
                identity={'name':str(names[0]),'distance':round(float(distances[0]),3),'threshold':round(im.threshold,3)}
        return {'pressure':p.tolist(),'metrics':pressure_metrics(p),'pose':pose,
                'pose_name':POSES[pose] if pose is not None else '模型未训练','model':best,
                'state':state,'motion_index':change,'regions':predicted,'region_names':REGIONS,
                'identity':identity,'airbags':simulate_airbags(p,elapsed,strength),
                'airbag_source':'单一调节示例；气囊编号参考布置图，传感器坐标为归一化示意映射，非标定控制映射',
                'sleep_note':'显示在床、静卧、体动和翻身；无睡眠分期标注，无法判定深睡/浅睡。'}

    def frame(self,person,action,frame):
        indices,i=self.select(person,action,frame)
        p=self.raw.pressure[i]
        result=self.infer(p,self.raw.pressure[indices[frame-1]] if frame else None)
        result.update({'person':person,'action':action,'frame':frame,'frames':len(indices),
                       'true_pose':int(self.raw.poses[i]),'source':'静态采集原始数据'})
        a=self.annotations()
        candidates=np.flatnonzero((a.people==person)&(a.actions==action))
        matches=[j for j in candidates if np.array_equal(a.pressure[j],p)]
        result['annotation']=a.boxes[matches[0]].tolist() if matches else None
        return result

    def replay(self,index,frame,strength):
        data=self.dynamic(index)
        if not 0<=frame<len(data):
            raise ValueError('动态帧号越界')
        seconds=70*frame/max(len(data)-1,1)
        result=self.infer(data[frame],data[frame-1] if frame else None,seconds,strength)
        result.update({'frame':frame,'frames':len(data),'seconds':seconds,'source':self.dynamic_paths[index].name,
                       'person':self.dynamic_paths[index].parent.name})
        return result

    def enroll(self,name,person,action):
        if not isinstance(name,str) or not 1<=len(name.strip())<=40:
            raise ValueError('注册名称需为 1–40 个字符')
        indices,_=self.select(person,action,0)
        model=self.get_model('identity_enrolled.joblib') or self.get_model('identity.joblib')
        if model is None:
            raise ValueError('请先运行训练')
        with self.lock:
            model.enroll(name.strip(),features(self.raw.pressure[indices[:5]]))
            path=MODELS/'identity_enrolled.joblib'
            tmp=path.with_suffix('.tmp')
            joblib.dump(model,tmp,compress=3);tmp.replace(path)
        return {'name':name.strip(),'templates_added':5,'note':'新增模板已保存；原始评估报告保持训练时快照，未重新评估。'}
