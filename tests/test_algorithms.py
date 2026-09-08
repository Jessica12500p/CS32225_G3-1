import numpy as np
import pytest
from src.alg.data import parse_txt, load_regions, split_groups, augment, canonical_action
from src.alg.features import features
from src.alg.regions import iou
from src.alg.telemetry import pressure_metrics, airbag_layout, motion_state, simulate_airbags
from src.alg.identity import IdentityMatcher


def test_parser_ignores_dynamic_labels_and_rejects_partial_frames(tmp_path):
    path=tmp_path/'dynamic.txt'
    path.write_text('1\n'+'\n'.join([','.join(['7']*24)]*44)+'\n\n2\n')
    assert parse_txt(path).shape==(1,44,24)
    path.write_text(','.join(['7']*24))
    with pytest.raises(ValueError):parse_txt(path)


def test_pressure_metrics_are_in_raw_units():
    p=np.zeros((44,24));p[10,5]=10;p[10,6]=20
    result=pressure_metrics(p)
    assert result['maximum']==20
    assert result['mean']==pytest.approx(30/1056)
    assert result['contact_mean']==15
    assert result['contact_points']==2
    assert result['center']==pytest.approx([17/3,10])
    assert pressure_metrics(np.zeros((44,24)))['center'] is None
    assert motion_state(np.zeros((44,24)))[0]=='离床'


def test_augmented_inputs_are_finite_and_keep_empty_sensor_points():
    p=np.zeros((3,44,24),dtype=np.float32);p[:,5:25,6:18]=20
    a=augment(p)
    assert np.all(a[p==0]==0)
    assert features(a).shape==features(p).shape
    assert np.isfinite(features(np.zeros_like(p))).all()
    with pytest.raises(ValueError):features(np.full_like(p,np.nan))


def test_mirror_families_cannot_cross_partitions():
    g=np.array([f'u:{canonical_action(action)}:{frame}' for action in (10,16,11,17) for frame in range(15)])
    train,test=split_groups(g)
    assert not set(g[train])&set(g[test])


def test_iou_does_not_count_background_as_correct():
    a=np.array([[[0,0,4,4]]*5],float)
    b=np.array([[[2,0,6,4]]*5],float)
    np.testing.assert_allclose(iou(a,b),1/3)
    np.testing.assert_allclose(iou(a,a),1)


def test_actual_region_parser_handles_na_lower_legs():
    ds=load_regions()
    assert ds.pressure.shape==(21570,44,24)
    assert ds.boxes.shape==(21570,5,4)
    assert np.isfinite(ds.boxes).all()


def test_layout_ids_and_simulated_right_side_have_no_measured_pressure():
    layout=airbag_layout()
    assert sorted(b['id'] for b in layout)==list(range(68))
    bags=simulate_airbags(np.ones((44,24))*20,10)
    assert all(b['pressure'] is None for b in bags if b['side']=='right')
    assert all(.15<=b['fill']<=.85 for b in bags)


def test_identity_enrollment_does_not_change_projection():
    rng=np.random.default_rng(2);x=rng.normal(size=(80,70));m=IdentityMatcher().fit(x,np.array(['A']*80))
    before=m.projection.components_.copy()
    m.enroll('B',x[:3]+30)
    np.testing.assert_array_equal(before,m.projection.components_)
    assert m.match(x[:1]+30)[0][0]=='B'
