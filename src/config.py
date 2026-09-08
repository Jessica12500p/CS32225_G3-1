from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data'
LOGS = ROOT / 'log'
RESULTS = ROOT / 'output'
MODELS = RESULTS / 'models'
SEED = 42
SHAPE = (44, 24)
POSES = ['仰卧', '俯卧', '左侧卧', '右侧卧']
REGIONS = ['肩部', '背部', '腰部', '臀部', '大腿部']

def ensure_dirs():
    for path in (LOGS, RESULTS, MODELS):
        path.mkdir(parents=True, exist_ok=True)
