"""Same YOLO checkpoint: full image vs tiled inference, native confidence and unified V3 evaluator."""
import argparse,json,sys,time,collections
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parent)]
from pcb_vlm.data.yolo_v3 import read,write
from pcb_vlm.provenance import sha256,source_identity,validate_partition
from pcb_vlm.geometry.tiling import generate_tiles,map_tile_box_to_global,nms_merge_predictions
from evaluate import evaluate_detections,compute_official_coco_eval,scope_ground_truth,is_valid_box

def yolo_objects(boxes,classes,scores,names,width,height):
    out=[]
    for box,cls,score in zip(boxes,classes,scores):
        if not is_valid_box(box,width,height) or int(cls)!=cls or not 0<=int(cls)<len(names) or not 0<=score<=1:raise ValueError('Invalid YOLO box/class/confidence')
        out.append({'label':names[int(cls)],'bbox_xyxy':list(box),'score':float(score)})
    return out

def confidence_subset(predictions,threshold=.25):return [{**r,'objects':[o for o in r['objects'] if o['score']>=threshold]} for r in predictions]

def evaluate_prediction_files(gt,preds,boards,threshold=.25):
    ids=[i['id'] for i in gt['images']];fixed=confidence_subset(preds,threshold)
    return {'primary_conf':threshold,'primary':evaluate_detections(gt,fixed,.5,ids),'ap_collection_conf':.001,'coco':compute_official_coco_eval(gt,preds,expected_image_ids=ids),'per_image':{str(i):evaluate_detections(gt,fixed,.5,[i]) for i in ids},'per_board':{b:evaluate_detections(gt,fixed,.5,[i for i in ids if boards[i]==b]) for b in sorted(set(boards.values()))}}

def run(plan_path,weights,mode):
    plan=read(plan_path);root=Path(plan['task_root']);from train_yolo_v3 import configure_runtime;configure_runtime(root)
    import torch,numpy as np
    from PIL import Image
    from ultralytics import YOLO
    for f,d in plan['input_sha256'].items():
        if sha256(f)!=d:raise ValueError('Input changed')
    if source_identity()!=plan['source_tree_sha256']:raise ValueError('Frozen code changed')
    output=root/f'inference_{mode}';output.mkdir(exist_ok=False);D=Path(plan['canonical_dataset']);gt=read(D/'coco_annotations.json');splits=read(D/'splits.json');manifest=read(D/'images_manifest.json');validate_partition(splits,manifest);ids=splits['splits']['val']['image_ids'];gt=scope_ground_truth(gt,ids);assert len(gt['annotations'])==759;images=[r for r in manifest if r['image_id'] in ids];boards={r['image_id']:r['physical_board_id'] for r in images};names=read(root/'yolo_data/provenance.json')['names'];model=YOLO(weights);assert list(model.names.values())==names;conf=plan['inference'];params=dict(imgsz=640,conf=conf['ap_collection_conf'],iou=conf['iou_nms'],max_det=conf['max_det_per_input'],device=0,half=False,rect=False,augment=False,agnostic_nms=False,verbose=False);torch.set_num_threads(2)
    warm=time.perf_counter();model.predict(np.zeros((640,640,3),dtype=np.uint8),**params);torch.cuda.synchronize();warm=time.perf_counter()-warm
    shapes=[];handle=model.predictor.model.model.register_forward_pre_hook(lambda module,args:shapes.append(list(args[0].shape)))
    predictions=[];all_tiles=[];started=time.perf_counter()
    with (output/'tile_predictions.jsonl').open('x',encoding='utf-8') as log:
        for im in images:
            if sha256(im['image_path'])!=im['sha256']:raise ValueError('Image hash changed')
            pil=Image.open(im['image_path']).convert('RGB');tiles=generate_tiles(im['width'],im['height'],640,.2) if mode=='tiled' else [{'tile_id':0,'bbox':[0,0,im['width'],im['height']],'offset_x':0,'offset_y':0}];objects=[];begin=time.perf_counter()
            for t in tiles:
                crop=pil.crop(t['bbox']);torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();tic=time.perf_counter();res=model.predict(crop,**params)[0];torch.cuda.synchronize();elapsed=time.perf_counter()-tic
                rows=yolo_objects(res.boxes.xyxy.cpu().tolist(),res.boxes.cls.cpu().tolist(),res.boxes.conf.cpu().tolist(),names,crop.width,crop.height)
                raw={'image_id':im['image_id'],'physical_board_id':im['physical_board_id'],'tile_id':t['tile_id'],'tile_bbox':t['bbox'],'actual_input_shape':shapes[-1],'objects':rows,'wall_s':elapsed,'native_speed_ms':res.speed,'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'cap_hit':len(rows)>=conf['max_det_per_input']};all_tiles.append(raw);log.write(json.dumps(raw)+'\n');log.flush()
                objects.extend({**o,'bbox_xyxy':map_tile_box_to_global(o['bbox_xyxy'],t['offset_x'],t['offset_y'],im['width'],im['height'])} for o in rows)
            merged=nms_merge_predictions(objects,.5);item={'image_id':im['image_id'],'objects':merged,'status':'success' if merged else 'no_objects_detected','is_truncated':False,'num_tiles':len(tiles),'nms_before':len(objects),'nms_after':len(merged),'wall_s':time.perf_counter()-begin};predictions.append(item);write(output/'predictions.json',predictions);print(mode,'image',im['image_id'],'tiles',len(tiles),'before',len(objects),'after',len(merged),flush=True)
    handle.remove();metrics=evaluate_prediction_files(gt,predictions,boards,conf['primary_conf']);metrics['runtime']={'total_wall_s':time.perf_counter()-started,'warmup_s':warm,'calls':len(all_tiles),'nms_removed':sum(p['nms_before']-p['nms_after'] for p in predictions),'max_det_per_input':conf['max_det_per_input'],'cap_hits':sum(r['cap_hit'] for r in all_tiles),'peak_allocated_mib':max(r['peak_allocated_mib'] for r in all_tiles)};write(output/'metrics.json',metrics);write(output/'ground_truth.json',gt);write(output/'run_manifest.json',{'completed':True,'plan_sha256':sha256(plan_path),'weights':str(weights),'weights_sha256':sha256(weights),'mode':mode,'params':params,'gt_sha256':sha256(D/'coco_annotations.json'),'val_ids':ids,'source_tree_sha256':source_identity(),'confidence_policy':'Primary P/R/F1 at .25; AP from .001 native real scores; no 1.0 score replacement.'});print(json.dumps(metrics['primary'],indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--weights',required=True);p.add_argument('--mode',choices=['whole','tiled'],required=True);a=p.parse_args();run(a.plan,a.weights,a.mode)
