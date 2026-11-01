import pandas as pd
import pytest
from mlb_portfolio.data import swings, batted_balls, era_pairs
from mlb_portfolio.models import temporal
from mlb_portfolio.models import pipeline
from sklearn.linear_model import LogisticRegression

def test_swing_definition():
    d=pd.DataFrame({'description':['swinging_strike','foul_tip','missed_bunt','called_strike'], 'game_type':['R']*4, 'sz_top':[3.5]*4,'sz_bot':[1.5]*4,'plate_z':[2.5]*4,'release_speed':[95]*4})
    s=swings(d)
    assert s.target.tolist()==[1,0]
    assert s.z_norm.tolist()==[.5,.5]

def test_batted_selection():
    d=pd.DataFrame({'description':['hit_into_play']*3,'game_type':['R']*3,'launch_speed':[100,None,90],'launch_angle':[30,30,0],'events':['home_run','home_run','single']})
    assert batted_balls(d).target.tolist()==[1,0]

def test_join_no_gap_and_no_name_matching():
    d=pd.DataFrame({'IDfg':[1,1,1,2],'Season':[2021,2022,2024,2022],'ERA':[3,4,5,6],'IP':[30]*4})
    p=era_pairs(d)
    assert p.loc[p.Season.eq(2021),'target'].iloc[0]==4
    assert p.loc[p.Season.eq(2022),'target'].isna().all()
    with pytest.raises(ValueError):
        era_pairs(pd.concat([d,d.iloc[[0]]]))

def test_baseball_innings_conversion():
    d=pd.DataFrame({'IDfg':[1],'Season':[2024],'ERA':[3.0],'IP':[20.2]})
    assert era_pairs(d).IP_innings.iloc[0]==pytest.approx(20+2/3)

def test_time_split_disjoint():
    d=pd.DataFrame({'game_date':['2025-04-30','2025-05-01','2025-05-15','2025-05-16']})
    a,b,c=temporal(d)
    assert [len(a),len(b),len(c)]==[1,2,1]
    assert not(set(a.index)&set(c.index))

def test_pipeline_unknown_category_and_train_only_imputation():
    train=pd.DataFrame({'x':[1.,3.,None,5.], 'hand':['L','R','L','R']})
    model=pipeline(['x'],['hand'],LogisticRegression()).fit(train,[0,1,0,1])
    assert model.named_steps['pre'].named_transformers_['num'].named_steps['impute'].statistics_[0]==3
    pred=model.predict_proba(pd.DataFrame({'x':[None], 'hand':['unknown']}))
    assert pred.shape==(1,2)

def test_era_time_split_uses_target_year():
    d=pd.DataFrame({'target_year':[2019,2022,2023,2024,2025]})
    a,b,c=temporal(d,True)
    assert a.target_year.max()==2022
    assert b.target_year.tolist()==[2023]
    assert c.target_year.tolist()==[2024,2025]
