"""Build independent YOLO data directly from canonical V3 COCO."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_vlm.data.yolo_v3 import build_dataset
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True);a=p.parse_args();print(json.dumps(build_dataset(a.dataset,a.output)['summary'],indent=2))
