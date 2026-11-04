"""Independently recompute YOLO labels, source pixels and complete GT coverage."""
import argparse,sys,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_vlm.data.yolo_v3 import audit_dataset
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit_dataset(a.data);Path(a.output).write_text(json.dumps(r,indent=2),encoding='utf-8');print(json.dumps(r,indent=2))
