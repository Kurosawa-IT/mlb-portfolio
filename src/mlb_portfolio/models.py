"""時間順の検証と、事前に限定したモデル比較実験。"""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss, accuracy_score, mean_absolute_error, mean_squared_error

def pipeline(num, cat, estimator):
    pre = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat)
    ])
    return Pipeline([('pre', pre), ('model', estimator)])

def temporal(df, era=False):
    if era:
        return df[df.target_year <= 2022], df[df.target_year == 2023], df[df.target_year.isin([2024, 2025])]
    date = pd.to_datetime(df.game_date)
    return df[date < '2025-05-01'], df[(date >= '2025-05-01') & (date < '2025-05-16')], df[date >= '2025-05-16']

def metrics(y, p, regression=False):
    if regression:
        return {'MAE': mean_absolute_error(y,p), 'RMSE': np.sqrt(mean_squared_error(y,p))}
    return {'ROC_AUC': roc_auc_score(y,p), 'PR_AUC_AP': average_precision_score(y,p), 'log_loss': log_loss(y,p), 'Brier': brier_score_loss(y,p), 'accuracy_0.5': accuracy_score(y,p>=.5)}

def experiments(df, task):
    era = task == 'era'
    train, val, test = temporal(df, era)
    if task == 'whiff':
        basic, cat = ['release_speed', 'plate_x', 'z_norm', 'balls', 'strikes'], ['pitch_type','p_throws','stand']
        extra = basic + ['speed_x_z', 'release_spin_rate', 'pfx_x', 'pfx_z']
    elif task == 'hr':
        basic, cat = ['launch_speed','launch_angle'], []
        extra = basic + ['speed_angle','angle2','speed2']
    else:
        basic, cat = ['ERA'], []
        extra = ['ERA','FIP','WHIP','K/9','BB/9','HR/9','Age','IP_innings','GS','G']
    configs = [('baseline_mean', basic, DummyRegressor() if era else DummyClassifier(strategy='prior')),
               ('ridge_basic' if era else 'logistic_basic', basic, Ridge(alpha=10) if era else LogisticRegression(max_iter=500)),
               ('ridge_extended' if era else 'logistic_extended', extra, Ridge(alpha=10) if era else LogisticRegression(max_iter=500)),
               ('rf_leaf20', extra, RandomForestRegressor(n_estimators=150,min_samples_leaf=20,max_depth=10,random_state=42,n_jobs=-1) if era else RandomForestClassifier(n_estimators=150,min_samples_leaf=20,max_depth=12,random_state=42,n_jobs=-1)),
               ('rf_leaf50', extra, RandomForestRegressor(n_estimators=150,min_samples_leaf=50,max_depth=8,random_state=42,n_jobs=-1) if era else RandomForestClassifier(n_estimators=150,min_samples_leaf=50,max_depth=10,random_state=42,n_jobs=-1))]
    rows, fitted = [], {}
    for name, nums, est in configs:
        m = pipeline(nums, cat, est).fit(train,train.target)
        p = m.predict(val) if era else m.predict_proba(val)[:,1]
        rows.append({'task':task,'model':name,'split':'validation','n_train':len(train),'n_eval':len(val), 'features':','.join(nums+cat), **metrics(val.target,p,era)})
        fitted[name] = m
    if era:
        rows.append({'task':task,'model':'previous_ERA','split':'validation','n_train':len(train),'n_eval':len(val),**metrics(val.target,val.ERA,True)})
    table = pd.DataFrame(rows)
    criterion = 'RMSE' if era else 'log_loss'
    winner = table.loc[table[criterion].idxmin(),'model']
    # テスト評価前にモデルを固定し、選択モデルと事前指定の基準だけを評価する。
    names = list(dict.fromkeys([winner,'baseline_mean'] + (['previous_ERA'] if era else [])))
    predictions = {}
    for name in names:
        p = test.ERA.to_numpy() if name == 'previous_ERA' else (fitted[name].predict(test) if era else fitted[name].predict_proba(test)[:,1])
        predictions[name] = p
        rows.append({'task':task,'model':name,'split':'test','n_train':len(train),'n_eval':len(test),**metrics(test.target,p,era)})
    return pd.DataFrame(rows), winner, fitted.get(winner), test, predictions

def cluster_bootstrap(test, predictions, winner, era=False, n_boot=400):
    """同一試合・投手内の依存を考慮し、対応付きの評価指標差を再標本化する。"""
    groups = test.IDfg if era else test.game_pk
    indices = {g: np.flatnonzero(groups.to_numpy()==g) for g in groups.unique()}
    keys = list(indices)
    rng = np.random.default_rng(42)
    baseline = 'previous_ERA' if era else 'baseline_mean'
    diffs=[]
    for _ in range(n_boot):
        ix = np.concatenate([indices[keys[j]] for j in rng.integers(0,len(keys),len(keys))])
        y = test.target.to_numpy()[ix]
        a, b = predictions[winner][ix], predictions[baseline][ix]
        diffs.append(np.sqrt(mean_squared_error(y,a))-np.sqrt(mean_squared_error(y,b)) if era else log_loss(y,a,labels=[0,1])-log_loss(y,b,labels=[0,1]))
    return {'comparison':f'{winner} minus {baseline}', 'metric':'RMSE' if era else 'log_loss', 'lower95':np.quantile(diffs,.025),'upper95':np.quantile(diffs,.975),'clusters':len(keys),'bootstrap':n_boot}
