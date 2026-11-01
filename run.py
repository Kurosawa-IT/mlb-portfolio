"""リポジトリルートからNotebookを生成し、順番に実行する。"""
from pathlib import Path
import argparse
import os
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.kernelspec import KernelSpecManager
import json
import asyncio
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--phase',choices=['whiff','hr','era','all'],default='all')
args=parser.parse_args()
os.environ['PYTHONPATH']=str(ROOT/'src')
os.environ['MPLBACKEND']='module://matplotlib_inline.backend_inline'
kernel_dir=ROOT/'.kernel'
kernel_dir.mkdir(exist_ok=True)
(kernel_dir/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],'display_name':'MLB portfolio','language':'python'}))
os.environ['JUPYTER_PATH']=str(ROOT/'.jupyter')
os.environ.setdefault('IPYTHONDIR',str(ROOT/'.jupyter'/'ipython'))
os.environ.setdefault('JUPYTER_CONFIG_DIR',str(ROOT/'.jupyter'/'config'))
os.environ.setdefault('JUPYTER_RUNTIME_DIR',str(ROOT/'.jupyter'/'runtime'))
Path(os.environ['JUPYTER_RUNTIME_DIR']).mkdir(parents=True,exist_ok=True)
import shutil
dest=ROOT/'.jupyter'/'kernels'/'mlb'
dest.mkdir(parents=True,exist_ok=True)
shutil.copyfile(kernel_dir/'kernel.json',dest/'kernel.json')

# 説明文とコードセルの定義を共通モジュールから読み込む
sys.path.insert(0, str(ROOT / 'src'))
from mlb_portfolio.notebook_content import cell_sources

for task in (['whiff','hr','era'] if args.phase=='all' else [args.phase]):
    # テーマごとに研究目的・対象定義・評価方法・結論を含むセルを作成する
    cells = [
        nbf.v4.new_markdown_cell(source) if kind == 'markdown' else nbf.v4.new_code_cell(source)
        for kind, source in cell_sources(task)
    ]
    nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'mlb','display_name':'MLB portfolio','language':'python'}})
    path=ROOT/'notebooks'/f'{["whiff","hr","era"].index(task)+1:02d}_{task}.ipynb'
    print('Executing',path,flush=True)
    try:
        NotebookClient(nb,timeout=1200,kernel_name='mlb',resources={'metadata':{'path':str(ROOT)}}).execute()
    finally:
        nbf.write(nb,path)
    print('Completed',task,flush=True)
