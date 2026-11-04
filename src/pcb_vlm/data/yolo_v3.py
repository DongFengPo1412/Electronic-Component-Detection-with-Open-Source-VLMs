"""Lossless V3 COCO -> YOLO labels with board provenance and explicit boundary supervision."""
import collections,json,math,random
from pathlib import Path
import yaml
from PIL import Image
from pcb_vlm.provenance import sha256,validate_partition
from pcb_vlm.geometry.tiling import generate_tiles

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')
def category_mapping(categories):
    rows=sorted(categories,key=lambda c:c['id'])
    if len({r['id'] for r in rows})!=len(rows) or len({r['name'] for r in rows})!=len(rows):raise ValueError('Duplicate category identity')
    return {r['id']:i for i,r in enumerate(rows)},[r['name'] for r in rows]

def xyxy_to_yolo(box,width,height):
    x1,y1,x2,y2=box
    if not all(math.isfinite(x) for x in box) or not (0<=x1<x2<=width and 0<=y1<y2<=height):raise ValueError('Illegal local box')
    return [(x1+x2)/2/width,(y1+y2)/2/height,(x2-x1)/width,(y2-y1)/height]

def yolo_to_xyxy(values,width,height):
    x,y,w,h=values
    if not all(math.isfinite(v) for v in values) or not (0<=x<=1 and 0<=y<=1 and 0<w<=1 and 0<h<=1):raise ValueError('Invalid YOLO values')
    return [(x-w/2)*width,(y-h/2)*height,(x+w/2)*width,(y+h/2)*height]

def clip_objects(annotations,region,mapping):
    x1,y1,x2,y2=region;out=[]
    for a in annotations:
        x,y,w,h=a['bbox'];l,t,r,b=max(x,x1),max(y,y1),min(x+w,x2),min(y+h,y2)
        if r<=l or b<=t:continue
        local=[l-x1,t-y1,r-x1,b-y1]; vis=(r-l)*(b-t)/(w*h)
        out.append({'annotation_id':a['id'],'source_object_id':a.get('source_object_id'),'coco_category_id':a['category_id'],'yolo_class_id':mapping[a['category_id']],'local_xyxy':local,'yolo_xywh':xyxy_to_yolo(local,x2-x1,y2-y1),'visibility':vis,'fully_visible':vis>=1-1e-9})
    return out

def source_regions(image,anns,mapping,size=640,overlap=.2):
    regions={tuple(t['bbox']):'grid' for t in generate_tiles(image['width'],image['height'],size,overlap)}
    full={o['annotation_id'] for box in regions for o in clip_objects(anns,box,mapping) if o['fully_visible']}
    for a in anns:
        if a['id'] in full:continue
        x,y,w,h=a['bbox']
        if w>size or h>size:box=(0,0,image['width'],image['height']);kind='whole_context_for_oversized'
        else:
            cw,ch=min(size,image['width']),min(size,image['height']);left=max(0,min(image['width']-cw,math.floor(x+w/2-cw/2)));top=max(0,min(image['height']-ch,math.floor(y+h/2-ch/2)));box=(left,top,left+cw,top+ch);kind='centered_for_full_visibility'
        regions.setdefault(box,kind)
        full.update(o['annotation_id'] for o in clip_objects(anns,box,mapping) if o['fully_visible'])
    return regions

def summarize(records,expected):
    ids={o['annotation_id'] for r in records for o in r['objects']};fully={o['annotation_id'] for r in records for o in r['objects'] if o['fully_visible']};counts=collections.Counter(o['yolo_class_id'] for r in records for o in r['objects']);boards=collections.Counter(r['physical_board_id'] for r in records)
    return {'records':len(records),'positive':sum(bool(r['objects']) for r in records),'negative':sum(not r['objects'] for r in records),'target_occurrences':sum(len(r['objects']) for r in records),'unique_gt':len(ids),'expected_unique_gt':len(expected),'fully_visible_unique_gt':len(fully),'missing_gt_ids':sorted(set(expected)-ids),'without_full_view_gt_ids':sorted(set(expected)-fully),'partial_occurrences':sum(not o['fully_visible'] for r in records for o in r['objects']),'under2px_occurrences':sum(min(o['local_xyxy'][2]-o['local_xyxy'][0],o['local_xyxy'][3]-o['local_xyxy'][1])<=2 for r in records for o in r['objects']),'class_occurrences':dict(counts),'class_unique_gt':{str(c):len({o['annotation_id'] for r in records for o in r['objects'] if o['yolo_class_id']==c}) for c in counts},'class_board_ids':{str(c):sorted({r['physical_board_id'] for r in records if any(o['yolo_class_id']==c for o in r['objects'])}) for c in counts},'board_records':dict(boards),'density':dict(collections.Counter('0' if not r['objects'] else '1-8' if len(r['objects'])<=8 else '9-32' if len(r['objects'])<=32 else '>32' for r in records))}

def build_dataset(dataset,output,size=640,overlap=.2):
    dataset=Path(dataset);output=Path(output);output.mkdir(parents=True,exist_ok=False);gt=read(dataset/'coco_annotations.json');split=read(dataset/'splits.json');manifest=read(dataset/'images_manifest.json');validate_partition(split,manifest);mapping,names=category_mapping(gt['categories']);all_records=[];summary={};by_image=collections.defaultdict(list)
    for a in gt['annotations']:by_image[a['image_id']].append(a)
    for part in ('train','val'):
        candidates=[]
        for im in manifest:
            if im['image_id'] not in split['splits'][part]['image_ids']:continue
            if sha256(im['image_path'])!=im['sha256']:raise ValueError('Source image changed')
            for region,kind in source_regions(im,by_image[im['image_id']],mapping,size,overlap).items():candidates.append({'image_entry':im,'tile_bbox':list(region),'kind':kind,'objects':clip_objects(by_image[im['image_id']],region,mapping)})
        positives=[r for r in candidates if r['objects']];negatives=[r for r in candidates if not r['objects']];random.Random(42).shuffle(negatives);kept=positives+negatives[:max(1,len(positives)//3)];kept.sort(key=lambda r:(r['image_entry']['image_id'],r['tile_bbox']))
        (output/f'images/{part}').mkdir(parents=True);(output/f'labels/{part}').mkdir(parents=True);records=[];current=None;pil=None
        for index,r in enumerate(kept):
            im=r.pop('image_entry');iid=im['image_id'];sid=f'{part}_{iid}_{index:05d}';ip=output/f'images/{part}/{sid}.png';lp=output/f'labels/{part}/{sid}.txt'
            if current!=iid:pil=Image.open(im['image_path']).convert('RGB');current=iid
            pil.crop(r['tile_bbox']).save(ip);width=r['tile_bbox'][2]-r['tile_bbox'][0];height=r['tile_bbox'][3]-r['tile_bbox'][1]
            text=''.join(str(o['yolo_class_id'])+' '+' '.join(f'{v:.12f}' for v in o['yolo_xywh'])+'\n' for o in r['objects']);lp.write_text(text,encoding='utf-8')
            records.append({**r,'id':sid,'image_id':iid,'board_folder':im['folder'],'physical_board_id':im['physical_board_id'],'split':part,'source_image_path':im['image_path'],'source_image_sha256':im['sha256'],'image_path':str(ip),'label_path':str(lp),'image_sha256':sha256(ip),'label_sha256':sha256(lp),'width':width,'height':height,'dataset_version':gt['info']['version']})
        expected={a['id'] for a in gt['annotations'] if a['image_id'] in split['splits'][part]['image_ids']};summary[part]=summarize(records,expected);summary[part]['negative_candidates']=len(negatives);summary[part]['discarded_background_regions']=len(negatives)-sum(not r['objects'] for r in records)
        if summary[part]['missing_gt_ids'] or summary[part]['without_full_view_gt_ids']:raise ValueError('Incomplete unique/full-view target coverage')
        all_records.extend(records)
    info={'dataset':str(dataset),'annotations_sha256':sha256(dataset/'coco_annotations.json'),'split_sha256':sha256(dataset/'splits.json'),'manifest_sha256':sha256(dataset/'images_manifest.json'),'coco_to_yolo':mapping,'names':names,'tile_size':size,'overlap':overlap,'boundary_policy':'Every positive-area intersection gets a full clipped label; no visibility or size deletion. Add centered crops or whole context so every original target also has a fully visible training view. No affine augmentation allowed to silently filter tiny boxes.','records':all_records,'summary':summary}
    write(output/'provenance.json',info);write(output/'data-audit-summary.json',summary);(output/'dataset.yaml').write_text(yaml.safe_dump({'path':str(output),'train':'images/train','val':'images/val','names':dict(enumerate(names))},sort_keys=False),encoding='utf-8');return info

def audit_dataset(output):
    output=Path(output);data=read(output/'provenance.json');source=Path(data['dataset']);gt=read(source/'coco_annotations.json');splits=read(source/'splits.json');manifest=read(source/'images_manifest.json');validate_partition(splits,manifest);mapping,names=category_mapping(gt['categories']);assert data['annotations_sha256']==sha256(source/'coco_annotations.json') and data['split_sha256']==sha256(source/'splits.json') and data['manifest_sha256']==sha256(source/'images_manifest.json');byid={r['image_id']:r for r in manifest};hashes={};seen=set();stats={};byann=collections.defaultdict(list)
    for a in gt['annotations']:byann[a['image_id']].append(a)
    cfg=yaml.safe_load((output/'dataset.yaml').read_text(encoding='utf-8'));assert cfg['names']==dict(enumerate(names)) and Path(cfg['path']).resolve()==output.resolve();assert {int(k):v for k,v in data['coco_to_yolo'].items()}==mapping and data['names']==names
    current=None;pil=None
    for r in data['records']:
        im=byid[r['image_id']];part=r['split'];assert part in ('train','val') and r['image_id'] in splits['splits'][part]['image_ids'] and r['physical_board_id']==im['physical_board_id'] and r['physical_board_id'] in splits['splits'][part]['boards'];key=(r['image_id'],tuple(r['tile_bbox']));assert key not in seen;seen.add(key)
        sp=im['image_path'];hashes.setdefault(sp,sha256(sp) if sp not in hashes else hashes[sp]);assert hashes[sp]==r['source_image_sha256']==im['sha256'];assert sha256(r['image_path'])==r['image_sha256'] and sha256(r['label_path'])==r['label_sha256']
        if current!=r['image_id']:pil=Image.open(sp).convert('RGB');current=r['image_id']
        assert Image.open(r['image_path']).convert('RGB').tobytes()==pil.crop(r['tile_bbox']).tobytes()
        expected=clip_objects(byann[r['image_id']],r['tile_bbox'],mapping);assert expected==r['objects'];lines=Path(r['label_path']).read_text().splitlines();assert len(lines)==len(expected)
        for line,obj in zip(lines,expected):
            values=line.split();assert int(values[0])==obj['yolo_class_id'] and 0<=int(values[0])<len(names);reconstructed=yolo_to_xyxy([float(x) for x in values[1:]],r['width'],r['height']);assert max(abs(a-b) for a,b in zip(reconstructed,obj['local_xyxy']))<1e-6
    for part in ('train','val'):
        expected={a['id'] for a in gt['annotations'] if a['image_id'] in splits['splits'][part]['image_ids']};stats[part]=summarize([r for r in data['records'] if r['split']==part],expected);assert not stats[part]['missing_gt_ids'] and not stats[part]['without_full_view_gt_ids']
    return {'passed':True,'summary':stats,'no_test_data_generated':True,'source_hashes_verified':len(hashes)}
