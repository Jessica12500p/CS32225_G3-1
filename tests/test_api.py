import pytest
from src.visual.app import create_app

@pytest.fixture(scope='module')
def client():
    return create_app().test_client()


def test_catalog_and_assets(client):
    catalog=client.get('/api/catalog').get_json()
    assert len(catalog['people'])==33
    assert len(catalog['images'])==12
    assert len(catalog['sessions'])>0
    assert client.get('/').status_code==200
    assert client.get('/api/images/raw-1').mimetype=='image/png'
    assert client.get('/static/app.js').status_code==200


def test_prediction_and_annotation_are_distinct(client):
    data=client.get('/api/frame?person=SAI&action=1&frame=0').get_json()
    assert data['annotation'] is not None
    assert len(data['regions'])==5
    assert data['pose'] in range(4)
    assert data['model']=='SVM-RBF'
    assert data['metrics']['maximum']==max(max(r) for r in data['pressure'])


def test_replay_actual_frames_and_boundaries(client):
    a=client.get('/api/replay?session=0&frame=0').get_json()
    b=client.get('/api/replay?session=0&frame=1').get_json()
    assert a['pressure']!=b['pressure']
    assert b['seconds']>a['seconds']
    assert len(a['airbags'])==68
    for query in ['frame=-1','session=-1','frame=999999','strength=nan','strength=2','frame=abc']:
        response=client.get('/api/replay?'+query)
        assert response.status_code==400
        assert 'error' in response.get_json()


def test_invalid_inputs_return_json(client):
    for url in ['/api/frame?person=missing','/api/frame?action=22','/api/frame?frame=-1','/api/images/missing']:
        response=client.get(url)
        assert response.status_code in (400,404)
        assert 'error' in response.get_json()
    assert client.post('/api/enroll',json={}).status_code==400


def test_report_and_export_match(client):
    data=client.get('/api/report').get_json()
    assert data['completed']
    assert not set(data['train_users']) & set(data['test_users'])
    assert data['posture']['models']['SVM-RBF']['accuracy']>.95
    assert client.get('/api/export').status_code==200
    assert len(client.get('/api/projection').get_json()['points'])>0


def test_enrollment_persists_to_separate_model(tmp_path,monkeypatch):
    import shutil
    import joblib
    import src.visual.service as module
    from src.config import MODELS
    from src.alg.features import features
    shutil.copyfile(MODELS/'identity.joblib',tmp_path/'identity.joblib')
    monkeypatch.setattr(module,'MODELS',tmp_path)
    service=module.MattressService()
    original=(tmp_path/'identity.joblib').read_bytes()
    result=service.enroll('验收用户','SAI',1)
    assert result['templates_added']==5
    assert (tmp_path/'identity.joblib').read_bytes()==original
    restored=joblib.load(tmp_path/'identity_enrolled.joblib')
    _,i=service.select('SAI',1,0)
    assert restored.predict(features(service.raw.pressure[i]))[0][0]=='验收用户'
