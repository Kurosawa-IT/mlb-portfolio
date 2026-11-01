"""Notebookで共用する探索的分析、統計モデル、評価処理。"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.proportion import proportion_confint
from sklearn.calibration import calibration_curve
from IPython.display import display
from .data import REPORTS, swings, batted_balls, era_pairs
from .models import experiments, temporal, cluster_bootstrap

sns.set_theme(style='whitegrid')

def savefig(name):
    plt.tight_layout()
    plt.savefig(REPORTS / 'figures' / f'{name}.png', dpi=140, bbox_inches='tight')
    plt.show()
    plt.close()

def rates(df, key):
    g=df.groupby(key,observed=True).target.agg(['size','sum','mean'])
    g['lower95'],g['upper95']=proportion_confint(g['sum'],g['size'],method='wilson')
    return g

def quality(raw, df, task):
    if task!='era':
        reg=raw[raw.game_type.eq('R')]
        pool=reg[reg.description.eq('hit_into_play')] if task=='hr' else reg[reg.description.isin(__import__('mlb_portfolio.data',fromlist=['SWINGS']).SWINGS)]
        audit=pd.DataFrame({'rows':[len(raw),len(reg),len(pool),len(df)],'stage':['deduplicated_source','regular_season','in_play' if task=='hr' else 'non_bunt_swings','eligible']})
        display(raw.groupby('game_type').size().rename('pitches'))
        print('Pitch keys duplicated:',raw.duplicated(['game_pk','at_bat_number','pitch_number']).sum())
    else:
        audit=pd.DataFrame({'rows':[len(raw),len(df),df.target.notna().sum()],'stage':['all_pitcher_seasons','year_t_20IP_and_t<=2024','observed_next_year']})
    display(audit); audit.to_csv(REPORTS/f'{task}_selection.csv',index=False)
    display(df.describe(include='all').T.head(25))
    miss=df.isna().mean().sort_values(ascending=False)
    miss.to_csv(REPORTS / f'{task}_missingness.csv')
    if task!='era':
        display(df.description.value_counts())
        display(pd.DataFrame([{'split':k,'n':len(v),'positive_rate':v.target.mean(),'start':str(v.game_date.min()),'end':str(v.game_date.max())} for k,v in zip(['train','validation','test'],temporal(df))]))
    else:
        display(df.groupby('target_year').target.agg(['size','count','mean']))

def whiff_eda(df):
    for key in ['pitch_type','speed_band','handedness']:
        if key=='speed_band': df[key]=pd.cut(df.release_speed,[0,80,85,90,95,100,110])
        if key=='handedness': df[key]=df.p_throws+' vs '+df.stand
        g=rates(df,key); display(g); g.to_csv(REPORTS / f'whiff_{key}_rates.csv')
        g['mean'].plot.bar(ylabel='Whiffs / swings',title=f'Whiff rate by {key}')
        savefig(f'whiff_{key}')
    d=df.assign(xbin=pd.cut(df.plate_x,np.arange(-2,2.01,.25)),zbin=pd.cut(df.z_norm,np.arange(-.5,1.76,.15)))
    agg=d.groupby(['zbin','xbin'],observed=True).target.agg(['mean','size'])
    sns.heatmap(agg['mean'].where(agg['size']>=30).unstack().iloc[::-1],vmin=0,vmax=.7,cmap='magma')
    plt.xlabel('Plate x (ft, catcher view)'); plt.ylabel('Normalized zone height'); plt.title('Whiff rate; bins with >=30 swings')
    savefig('whiff_location')

def hr_eda(df):
    sns.histplot(data=df,x='launch_speed',y='launch_angle',bins=45,cbar=True)
    plt.xlabel('Exit velocity (mph)'); plt.ylabel('Launch angle (deg)'); savefig('hr_distribution')
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    for ax,key in zip(axes,['launch_speed','launch_angle']):
        sns.histplot(data=df,x=key,hue='target',stat='density',common_norm=False,element='step',fill=False,ax=ax)
    savefig('hr_class_distributions')
    d=df.assign(evbin=pd.cut(df.launch_speed,np.arange(30,126,5)),labin=pd.cut(df.launch_angle,np.arange(-90,91,10)))
    g=d.groupby(['labin','evbin'],observed=True).target.agg(['mean','size','sum'])
    g['lower95'],g['upper95']=proportion_confint(g['sum'],g['size'],method='wilson')
    g.to_csv(REPORTS/'hr_regions.csv')
    display(g[g['size']>=30].sort_values('mean',ascending=False).head(12))
    sns.heatmap(g['mean'].where(g['size']>=30).unstack().iloc[::-1],cmap='magma',vmin=0,vmax=1)
    plt.xlabel('Exit velocity (mph)'); plt.ylabel('Launch angle (deg)'); plt.title('HR rate; bins with >=30 balls'); savefig('hr_regions')

def inference(df,task):
    # 統計推定は訓練データに限定し、標準誤差では同一試合内の依存を考慮する。
    train=temporal(df)[0].copy()
    formula = ('target ~ release_speed + plate_x + I(plate_x**2) + z_norm + I(z_norm**2) + C(pitch_type) + C(stand)*C(p_throws) + balls + strikes + release_speed:z_norm' if task=='whiff' else 'target ~ launch_speed + launch_angle + I(launch_speed**2) + I(launch_angle**2) + launch_speed:launch_angle')
    cols=['target','game_pk']+(['release_speed','plate_x','z_norm','pitch_type','stand','p_throws','balls','strikes'] if task=='whiff' else ['launch_speed','launch_angle'])
    train=train.dropna(subset=cols)
    if task=='whiff':
        counts=train.pitch_type.value_counts(); train=train[train.pitch_type.isin(counts[counts>=100].index)]
    if task=='hr':
        # 二次特徴量を中心化・尺度調整し、確率がほぼ0の領域でも安定して推定する。
        # Logitのスコア計算を使い、GLMの共分散計算で生じた欠損を回避する。
        train['ev10']=(train.launch_speed-90)/10
        train['la10']=(train.launch_angle-20)/10
        fit=smf.logit('target ~ ev10 + la10 + I(ev10**2) + I(la10**2) + ev10:la10',train).fit(disp=False,maxiter=200,cov_type='cluster',cov_kwds={'groups':train.game_pk})
    else:
        fit=smf.glm(formula,train,family=sm.families.Binomial()).fit(cov_type='cluster',cov_kwds={'groups':train.game_pk})
    ci=fit.conf_int()
    table=pd.DataFrame({'coefficient':fit.params,'lower95':ci[0],'upper95':ci[1],'p_value':fit.pvalues})
    if not np.isfinite(table.to_numpy()).all():
        raise ValueError('Nonfinite inferential coefficients or uncertainty estimates')
    display(table)
    table.to_csv(REPORTS/f'{task}_inference.csv')
    print('Training-only exploratory associations; game-cluster robust SE; no causal interpretation or multiple-testing correction.')

def evaluate(df,task):
    table,winner,model,test,preds=experiments(df,task)
    table.to_csv(REPORTS/f'{task}_experiments.csv',index=False)
    display(table)
    print('Validation-selected:',winner)
    ci=cluster_bootstrap(test,preds,winner,task=='era')
    display(pd.DataFrame([ci])); (REPORTS/f'{task}_bootstrap.json').write_text(json.dumps(ci,indent=2),encoding='utf-8')
    if task=='era':
        # 検証で選んだモデルは変更せず、固定した平均予測との差も評価する。
        ci_mean=cluster_bootstrap(test,{winner:preds[winner],'previous_ERA':preds['baseline_mean']},winner,True)
        ci_mean['comparison']=f'{winner} minus training_mean'
        display(pd.DataFrame([ci_mean]));(REPORTS/'era_mean_bootstrap.json').write_text(json.dumps(ci_mean,indent=2),encoding='utf-8')
    if task!='era':
        for name,p in preds.items():
            obs,prob=calibration_curve(test.target,p,n_bins=10,strategy='quantile')
            plt.plot(prob,obs,'o-',label=name)
        plt.plot([0,1],[0,1],'--',color='gray'); plt.xlabel('Predicted probability'); plt.ylabel('Observed fraction'); plt.legend(); savefig(f'{task}_calibration')
    else:
        out=test[['IDfg','Name','Season','target_year','IP','GS','ERA','target']].copy()
        out['prediction']=preds[winner];out['error']=out.prediction-out.target
        display(out.assign(abs_error=out.error.abs()).nlargest(15,'abs_error'))
        out.to_csv(REPORTS/'era_test_predictions.csv',index=False)
        out['role']=np.where(out.GS>=10,'10+ starts','<10 starts')
        out['ip_band']=pd.cut(out.IP,[19,50,100,150,300])
        for name,keys in [('year_role',['target_year','role']),('ip_band',['ip_band'])]:
            g=out.groupby(keys,observed=True).error.agg(n='size',MAE=lambda s:s.abs().mean(),RMSE=lambda s:np.sqrt((s**2).mean()))
            display(g);g.to_csv(REPORTS/f'era_errors_{name}.csv')
        fig,axes=plt.subplots(1,2,figsize=(12,4))
        for ax in axes:
            ax.scatter(out.target,out.prediction,alpha=.3);ax.set_xlabel('Next-year ERA');ax.set_ylabel('Predicted ERA')
        axes[0].set_title('All test observations')
        axes[1].set_xlim(0,12);axes[1].set_ylim(0,12);axes[1].plot([0,12],[0,12],'--',color='gray');axes[1].set_title('Detail: ERA 0–12')
        savefig('era_predictions')
    return table

def era_eda(raw,pairs):
    selected=pairs.dropna(subset=['target']).copy()
    attr=pairs.assign(returned=pairs.target.notna()).groupby('Season').agg(eligible=('IDfg','size'),returned=('returned','sum'))
    attr['absent_next_year']=attr.eligible-attr.returned
    display(attr);attr.to_csv(REPORTS/'era_attrition.csv')
    display(pairs.groupby(pairs.target.notna())[['ERA','IP','Age']].mean())
    cols=['ERA','FIP','WHIP','K/9','BB/9','HR/9','IP']
    sns.heatmap(raw[cols].corr(method='spearman'),annot=True,fmt='.2f',vmin=-1,vmax=1,cmap='vlag');savefig('era_correlations')
    sns.boxplot(data=raw[raw.IP>=20],x='Season',y='ERA',showfliers=False);savefig('era_years')
    plt.scatter(selected.ERA,selected.target,alpha=.15);plt.xlim(0,15);plt.ylim(0,15);plt.xlabel('Year t ERA');plt.ylabel('Year t+1 ERA');savefig('era_year_to_year')
    # 訓練データだけで回帰し、投手内の依存を考慮した区間で関連を確認する。
    tr=temporal(selected,True)[0].dropna(subset=['ERA','FIP','WHIP','K/9','BB/9','Age'])
    tr=tr.rename(columns={'K/9':'K9','BB/9':'BB9'})
    fit=smf.ols('target ~ ERA + FIP + WHIP + K9 + BB9 + Age',tr).fit(cov_type='cluster',cov_kwds={'groups':tr.IDfg})
    ci=fit.conf_int();tab=pd.DataFrame({'coefficient':fit.params,'lower95':ci[0],'upper95':ci[1],'p_value':fit.pvalues});display(tab);tab.to_csv(REPORTS/'era_inference.csv')
    return selected
