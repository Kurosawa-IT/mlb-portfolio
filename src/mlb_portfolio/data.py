"""データ取得とキャッシュ管理。生データはGit管理対象に含めない。"""
from pathlib import Path
import json
import hashlib
import os
from datetime import datetime, timezone
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / 'data' / 'raw'
REPORTS = ROOT / 'reports'
os.environ.setdefault('PYBASEBALL_CACHE', str(RAW / 'pybaseball_cache'))

def record(**kwargs):
    REPORTS.mkdir(exist_ok=True)
    with (REPORTS / 'acquisition.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps({'utc': datetime.now(timezone.utc).isoformat(), **kwargs}, ensure_ascii=False) + '\n')

def cached(name, loader):
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / f'{name}.parquet'
    if p.exists():
        return pd.read_parquet(p)
    try:
        df = loader()
        if df.empty:
            raise ValueError('Empty response')
        df.to_parquet(p, index=False)
        record(dataset=name, rows=len(df), columns=list(df.columns), sha256=hashlib.sha256(p.read_bytes()).hexdigest(), status='ok')
        return df
    except Exception as exc:
        record(dataset=name, status='error', error=repr(exc))
        raise

def statcast_data():
    from pybaseball import statcast
    # 週ごとに保存し、取得失敗時は未取得の期間から再開できるようにする。
    frames = []
    for start in pd.date_range('2025-04-01', '2025-05-31', freq='7D'):
        end = min(start + pd.Timedelta(days=6), pd.Timestamp('2025-05-31'))
        a, b = start.strftime('%Y-%m-%d'), end.strftime('%Y-%m-%d')
        frames.append(cached(f'statcast_{a}_{b}', lambda: statcast(a, b, parallel=True)))
    df = pd.concat(frames, ignore_index=True)
    return df.drop_duplicates(['game_pk', 'at_bat_number', 'pitch_number'])

def pitching_data():
    from pybaseball import pitching_stats
    def fetch(y):
        try:
            df = pitching_stats(y, y, qual=0)
            if df.empty:
                raise ValueError('pybaseball legacy HTML returned no rows')
            return df
        except Exception as exc:
            record(dataset=f'fangraphs_{y}', status='fallback', error=repr(exc))
            import requests
            from bs4 import BeautifulSoup
            url = 'https://www.fangraphs.com/api/leaders/major-league/data'
            r = requests.get(url, params={'age':'','pos':'all','stats':'pit','lg':'all','qual':'0','type':'8','season':y,'season1':y,'ind':'1','team':'0','pageitems':'2000','pagenum':'1'}, timeout=90)
            r.raise_for_status()
            payload = r.json()
            df = pd.DataFrame(payload['data'])
            record(dataset=f'fangraphs_{y}', status='fallback_response', url=r.url, response_keys=list(payload), rows=len(df))
            if len(df) >= 2000:
                raise ValueError('Pagination required')
            if 'IDfg' not in df:
                df['IDfg'] = df.Name.str.extract(r'playerid=(\d+)')[0].astype(int)
            df['Name'] = df.Name.map(lambda s: BeautifulSoup(s,'html.parser').get_text())
            return df
    return pd.concat([cached(f'fangraphs_{y}', lambda y=y: fetch(y)) for y in range(2018, 2026)], ignore_index=True)

SWINGS = ['swinging_strike', 'swinging_strike_blocked', 'foul', 'foul_tip', 'hit_into_play', 'foul_pitchout', 'swinging_pitchout']
WHIFFS = ['swinging_strike', 'swinging_strike_blocked', 'swinging_pitchout']

def swings(df):
    out = df.loc[df.description.isin(SWINGS) & df.game_type.eq('R')].copy()
    out['target'] = out.description.isin(WHIFFS).astype(int)
    # 打者別ゾーンで高さを正規化し、欠損は後段のPipelineで補完する。
    span = out.sz_top - out.sz_bot
    out['z_norm'] = (out.plate_z - out.sz_bot) / span.where(span > 0)
    out['speed_x_z'] = out.release_speed * out.z_norm
    return out

def batted_balls(df):
    out = df.loc[df.game_type.eq('R') & df.description.eq('hit_into_play')].dropna(subset=['launch_speed', 'launch_angle', 'events']).copy()
    out['target'] = out.events.eq('home_run').astype(int)
    out['speed_angle'] = out.launch_speed * out.launch_angle
    out['angle2'] = out.launch_angle ** 2
    out['speed2'] = out.launch_speed ** 2
    return out

def era_pairs(df, min_ip=20):
    if df.duplicated(['IDfg', 'Season']).any():
        raise ValueError('Duplicate IDfg/Season: aggregate traded-player rows before joining')
    nxt = df[['IDfg', 'Season', 'ERA']].rename(columns={'ERA': 'target'})
    nxt['Season'] -= 1
    out = df.merge(nxt, on=['IDfg', 'Season'], how='left', validate='one_to_one')
    out['target_year'] = out.Season + 1
    # IPの小数部.1/.2は1アウト・2アウトを表すため、1/3・2/3回に変換する。
    out['IP_innings'] = out.IP // 1 + ((out.IP % 1) * 10).round() / 3
    # 選択閾値は整数20回なので、元のIP表記でも大小判定は変わらない。
    return out.loc[(out.IP >= min_ip) & (out.Season <= 2024)].copy()
