"""First-tile labels manually verified against the supplied PNG page headings."""
from src.config import DATA, POSES


def image_catalog():
    result=[]
    # Only these inspected pages are claimed to match a person/posture.
    for person,pages in [('SAI',[1,35,3,16]),('dgs',[47,80,50,60])]:
        for pose,page in enumerate(pages):
            result.append({'id':f'raw-{page}','person':person,'pose':pose,'pose_name':POSES[pose],
                           'kind':'raw','page':page,'file':f'睡姿原始数据部分热力图/origin_heatmap_page_{page:03d}.png',
                           'note':'标签对应页面首图；同页可能含其他动作。'})
    for pose,page in enumerate([1,37,3,15]):
        result.append({'id':f'region-{page}','person':'hpy','pose':pose,'pose_name':POSES[pose],
                       'kind':'region','page':page,'file':f'区域划分数据部分热力图/page_{page:04d}.png',
                       'note':'提供的区域标注图，不是当前模型预测。'})
    return [item for item in result if (DATA/'heapmap(Partial)'/item['file']).is_file()]
