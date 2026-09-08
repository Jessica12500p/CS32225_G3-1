"""Measured pressure metrics and one explicitly simulated airbag scenario."""
import numpy as np


def pressure_metrics(pressure, threshold=5.):
    p = np.asarray(pressure,dtype=float)
    contact = p > threshold
    total = p.sum()
    yy,xx = np.indices(p.shape)
    return {'maximum':float(p.max()),'mean':float(p.mean()),
            'contact_mean':float(p[contact].mean()) if contact.any() else 0.,
            'contact_index':float(contact.mean()),'contact_points':int(contact.sum()),
            'threshold':threshold,'total':float(total),
            'center': [float((xx*p).sum()/total),float((yy*p).sum()/total)] if total else None,
            'unit':'原始读数'}


def motion_state(p, previous=None, pose=None, previous_pose=None):
    if np.max(p)<=5:
        return '离床',0.
    if previous is None:
        return '在床 · 首帧',0.
    change = float(np.abs(p-previous).sum()/max(previous.sum(),1))
    if pose != previous_pose and change>.15:
        return '翻身',change
    return ('体动' if change>.12 else '静卧'),change


def airbag_layout():
    rows_left = [[32,33,34,35,43,14],[24,25,26,27,4,5],[36,37,38,39,16,17],
                 [28,29,30,31,8,9],[20,21,22,23,0,1]]
    rows_right = [[15,67,56,57,58,59],[6,7,48,49,50,51],[18,19,60,61,62,63],
                  [10,11,52,53,54,55],[2,3,44,45,46,47]]
    result=[]
    for side,rows,bars,foot in [('left',rows_left,[40,41,42],12),('right',rows_right,[64,65,66],13)]:
        for iy,row in enumerate(rows):
            for ix,number in enumerate(row):
                result.append({'id':number,'side':side,'x':(ix+.5)/6,'y':(.36+iy*.107),'kind':'circle'})
        for i,number in enumerate(bars):
            result.append({'id':number,'side':side,'x':.5,'y':.045+i*.1,'kind':'bar'})
        result.append({'id':foot,'side':side,'x':.5,'y':.96,'kind':'bar'})
    return result


def simulate_airbags(p, elapsed=0., strength=.5):
    result=[]
    for bag in airbag_layout():
        x=min(23,int(bag['x']*24));y=min(43,int(bag['y']*44))
        sensor=float(p[y,x]) if bag['side']=='left' else None
        # One bounded illustrative control response, not measured physical pressure.
        target = float(np.clip(.5+strength*(.4-(sensor or 0)/max(float(p.max()),1)),.15,.85))
        fill = .5+(target-.5)*(1-np.exp(-max(elapsed,0)/6))
        result.append({**bag,'sensor':[y,x] if sensor is not None else None,'pressure':sensor,
                       'fill':float(fill),'target':target,'support_mm':round(18+32*fill,1)})
    return result
