"""Build V3 from every XML object; preserve legacy image/split identity without overwrites."""
import argparse,collections,copy,json,math,os,sys,time,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_vlm.data.raw_labels import classify_raw_name
from pcb_vlm.provenance import sha256,physical_id,validate_partition,code_identity,source_identity

def write_json(path,data):
    with Path(path).open('x',encoding='utf-8') as stream: json.dump(data,stream,indent=2,ensure_ascii=False)
def reconcile(old, ledger):
    lookup=collections.defaultdict(list)
    for row in ledger:
        if row['bbox_xyxy'] is not None:
            lookup[(row['image_id'],tuple(row['bbox_xyxy']))].append(row)
    result=[]; matched=set()
    for ann in old['annotations']:
        box=ann.get('bbox_xyxy') or [ann['bbox'][0],ann['bbox'][1],ann['bbox'][0]+ann['bbox'][2],ann['bbox'][1]+ann['bbox'][3]]
        sources=lookup[(ann['image_id'],tuple(box))]
        if len(sources)!=1: raise ValueError(f'Ambiguous old annotation source: {ann["id"]}: {len(sources)}')
        row=sources[0]; key=row['source_object_id']
        if key in matched: raise ValueError(f'Duplicate old source annotation: {key}')
        matched.add(key)
        status=('unchanged' if ann['category_id']==row.get('category_id') else 'relabeled') if row['decision']=='keep' else ('ambiguous' if row['decision']=='ambiguous' else 'removed')
        result.append(dict(status=status,old_annotation_id=ann['id'],source_object_id=key,image_id=ann['image_id'],old_category_id=ann['category_id'],new_category_id=row.get('category_id'),raw_name=row['raw_name'],reason=row['reason']))
    for row in ledger:
        if row['source_object_id'] not in matched and row['decision'] in ('keep','ambiguous'):
            result.append(dict(status='added' if row['decision']=='keep' else 'ambiguous',old_annotation_id=None,source_object_id=row['source_object_id'],image_id=row['image_id'],old_category_id=None,new_category_id=row.get('category_id'),raw_name=row['raw_name'],reason=row['reason']))
    return result

def rebuild(old_dir, raw_dir, out_dir, policy_path):
    old_dir,raw_dir,out_dir=map(Path,(old_dir,raw_dir,out_dir))
    if out_dir.exists(): raise FileExistsError(out_dir)
    out_dir.mkdir(parents=True)
    policy=yaml.safe_load(Path(policy_path).read_text(encoding='utf-8'))
    old=json.loads((old_dir/'coco_annotations.json').read_text(encoding='utf-8')); manifest=json.loads((old_dir/'images_manifest.json').read_text(encoding='utf-8')); splits=json.loads((old_dir/'splits.json').read_text(encoding='utf-8')); classes=old['categories']; names={c['name']:c['id'] for c in classes}
    ledger=[]; annotations=[]; xml_sources=[]
    for image in manifest:
        path=Path(image['image_path']); folder=image['folder']; board=physical_id(folder)
        if sha256(path)!=image['sha256']: raise ValueError(f'Source image hash mismatch: {path}')
        with Image.open(path) as pic:
            if pic.size!=(image['width'],image['height']): raise ValueError('Image dimension mismatch')
        xml=raw_dir/folder/(folder+'.xml'); xhash=sha256(xml); root=ET.parse(xml).getroot(); objects=root.findall('object'); xml_sources.append(dict(path=str(xml),sha256=xhash,image_id=image['image_id'],object_count=len(objects)))
        for index,obj in enumerate(objects):
            raw_name=obj.findtext('name'); verdict=classify_raw_name(raw_name,policy); box=None; geometry_error=None
            try:
                box=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
                if not all(math.isfinite(v) for v in box) or not (0<=box[0]<box[2]<=image['width'] and 0<=box[1]<box[3]<=image['height']): raise ValueError('invalid original bbox')
            except (ValueError,TypeError) as exc: geometry_error=str(exc)
            row=dict(source_object_id=f'{folder}:{index}',xml_path=str(xml),xml_sha256=xhash,object_index=index,image_id=image['image_id'],board_folder=folder,physical_board_id=board,raw_name=raw_name,raw_bbox=box,bbox_xyxy=box,**verdict)
            if geometry_error:
                row['geometry_error']=geometry_error
                if row['decision']=='keep': row.update(decision='ambiguous',reason='invalid_target_geometry')
            if row['decision']=='keep':
                cid=names[row['category']]; row['category_id']=cid; row['new_annotation_id']=len(annotations)+1; x,y,X,Y=box
                annotations.append(dict(id=row['new_annotation_id'],image_id=image['image_id'],category_id=cid,category_name=row['category'],bbox=[x,y,X-x,Y-y],bbox_xyxy=box,area=(X-x)*(Y-y),iscrowd=0,board_group_id=board,source_object_id=row['source_object_id'],xml_path=str(xml),xml_sha256=xhash,object_index=index,raw_class=row['raw_class']))
            ledger.append(row)
        image.update(physical_board_id=board,board_group_id=board,num_components=sum(a['image_id']==image['image_id'] for a in annotations))
    diff=reconcile(old,ledger)
    coco={**old,'info':{**old['info'],'version':'3.0-semantic','date_created':time.strftime('%Y-%m-%d'),'semantic_policy_sha256':sha256(policy_path)},'annotations':annotations}
    for name,part in splits['splits'].items(): part['num_annotations']=sum(a['image_id'] in part['image_ids'] for a in annotations)
    splits['metadata'].update(dataset_version='3.0-semantic',total_annotations=len(annotations),source_split_sha256=sha256(old_dir/'splits.json'))
    splits['metadata'].pop('split_hash',None)
    import hashlib
    splits['metadata']['split_hash']=hashlib.sha256(json.dumps(splits['splits'],sort_keys=True).encode()).hexdigest()
    validate_partition(splits,manifest)
    for name in ('train','val','test'):
        original=json.loads((old_dir/'splits.json').read_text(encoding='utf-8'))['splits'][name]
        for key in ('boards','image_ids','image_folders'):
            if splits['splits'][name][key]!=original[key]: raise ValueError('Split identity changed')
    write_json(out_dir/'coco_annotations.json',coco); write_json(out_dir/'images_manifest.json',manifest); write_json(out_dir/'splits.json',splits)
    write_json(out_dir/'raw_objects.json',ledger); write_json(out_dir/'xml_sources.json',xml_sources); write_json(out_dir/'annotation_diff.json',diff); write_json(out_dir/'semantic_policy.json',policy)
    summary={'version':'3.0-semantic','raw_objects':len(ledger),'decisions':dict(collections.Counter(r['decision'] for r in ledger)),'raw_class_counts':dict(collections.Counter(r['raw_class'] for r in ledger)),'old_annotations':len(old['annotations']),'new_annotations':len(annotations),'diff_counts':dict(collections.Counter(r['status'] for r in diff)),'ambiguous':[r['source_object_id'] for r in ledger if r['decision']=='ambiguous'],'policy_sha256':sha256(policy_path),'old_coco_sha256':sha256(old_dir/'coco_annotations.json'),'old_splits_sha256':sha256(old_dir/'splits.json'),'git_commit':code_identity(),'source_tree_sha256':source_identity(),'files_sha256':{f.name:sha256(f) for f in out_dir.glob('*.json')}}
    write_json(out_dir/'dataset_version.json',summary)
    return summary

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--old-dir',required=True); ap.add_argument('--raw-dir',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--policy',default=str(Path(__file__).resolve().parents[1]/'configs/semantic_policy_v3.yaml')); args=ap.parse_args(); print(json.dumps(rebuild(args.old_dir,args.raw_dir,args.output_dir,args.policy),indent=2))
if __name__=='__main__': main()
