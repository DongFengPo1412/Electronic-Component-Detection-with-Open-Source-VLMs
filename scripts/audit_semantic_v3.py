"""Independent V3 XML coverage and SFT pixel/geometry/label audit, CPU only."""
import argparse,collections,json,math,shlex,sys,xml.etree.ElementTree as ET
from pathlib import Path
import yaml
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_vlm.provenance import sha256,validate_partition,validate_samples
from pcb_vlm.geometry.coordinate_protocol import parse_response

def audit_dataset(folder):
    folder=Path(folder); load=lambda n:json.loads((folder/n).read_text(encoding='utf-8'))
    gt=load('coco_annotations.json'); manifest=load('images_manifest.json'); splits=load('splits.json'); ledger=load('raw_objects.json'); policy=load('semantic_policy.json'); by_source={r['source_object_id']:r for r in ledger}; by_ann={a['source_object_id']:a for a in gt['annotations']}
    validate_partition(splits,manifest); expected=set(); ignored=set(); ambiguous=[]; count=0; source_hashes=[]
    for image in manifest:
        if sha256(image['image_path'])!=image['sha256']: raise ValueError('Source hash mismatch')
        xml=Path(image['image_path']).with_suffix('.xml'); xhash=sha256(xml); source_hashes.append(image['sha256'])
        for index,obj in enumerate(ET.parse(xml).getroot().findall('object')):
            key=f'{image["folder"]}:{index}'; row=by_source[key]; count+=1
            if row['xml_sha256']!=xhash or row['raw_name']!=obj.findtext('name'): raise ValueError('Ledger XML content mismatch')
            try: kind=' '.join(shlex.split(obj.findtext('name'))[0].lower().split())
            except (ValueError,TypeError,IndexError): kind=None
            coords=[float(obj.findtext('bndbox/'+k)) for k in ('xmin','ymin','xmax','ymax')]
            if kind in policy['retained_raw_classes']:
                if not all(math.isfinite(v) for v in coords) or not (0<=coords[0]<coords[2]<=image['width'] and 0<=coords[1]<coords[3]<=image['height']): ambiguous.append(key); continue
                expected.add(key); ann=by_ann[key]; name=next(c['name'] for c in gt['categories'] if c['id']==ann['category_id'])
                if name!=policy['retained_raw_classes'][kind] or ann['bbox_xyxy']!=coords or row['decision']!='keep': raise ValueError('Annotation semantic/geometry mismatch')
            elif kind in policy['ignored_raw_classes']:
                ignored.add(key)
                if row['decision']!='ignore' or key in by_ann: raise ValueError('Ignored object entered GT')
            else: ambiguous.append(key)
    if count!=len(ledger) or len(by_source)!=len(ledger) or expected!=set(by_ann) or ambiguous: raise ValueError('Bidirectional raw/XML coverage failed')
    if len(set(source_hashes))!=len(source_hashes): raise ValueError('Duplicate image contents')
    statistics={}
    for cat in gt['categories']:
        anns=[a for a in gt['annotations'] if a['category_id']==cat['id']]; ids={a['image_id'] for a in anns}; boards={i['physical_board_id'] for i in manifest if i['image_id'] in ids}
        statistics[cat['name']]={'annotations':len(anns),'images':len(ids),'physical_boards':len(boards)}
    return {'passed':True,'raw_objects':count,'retained':len(expected),'ignored':len(ignored),'ambiguous':ambiguous,'statistics':statistics,'partitions':{n:{'images':len(s['image_ids']),'boards':len(s['boards']),'annotations':s['num_annotations']} for n,s in splits['splits'].items()},'per_board':dict(collections.Counter(a['board_group_id'] for a in gt['annotations']))}

def audit_sft(folder,sft_dir,tokenizer_path):
    folder,sft_dir=Path(folder),Path(sft_dir); gt=json.loads((folder/'coco_annotations.json').read_text(encoding='utf-8')); ids={a['id']:a for a in gt['annotations']}; labels={c['id']:c['name'] for c in gt['categories']}
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(tokenizer_path,local_files_only=True)
    summary={}; used_sources=set(); last_path=None; source=None
    for name in ('train','val'):
        samples=json.loads((sft_dir/f'{name}_sft.json').read_text(encoding='utf-8')); boards=validate_samples(samples,name,folder/'splits.json'); seen=set(); lengths=[]; counts=collections.Counter(); density=collections.Counter(); negatives=0
        for r in samples:
            region=r['tile_bbox']; key=(r['image_id'],tuple(region))
            if key in seen: raise ValueError('Duplicate source region')
            seen.add(key)
            if last_path!=r['source_image_path']:
                source=Image.open(r['source_image_path']).convert('RGB'); last_path=r['source_image_path']
            actual=Image.open(r['image']).convert('RGB')
            if source.crop(region).tobytes()!=actual.tobytes() or actual.size!=(r['width'],r['height']): raise ValueError('SFT crop pixel mismatch')
            expected=[]; any_intersection=False
            for a in gt['annotations']:
                if a['image_id']!=r['image_id']: continue
                x,y,w,h=a['bbox']; l,t,R,b=max(x,region[0]),max(y,region[1]),min(x+w,region[2]),min(y+h,region[3]); area=max(0,R-l)*max(0,b-t)
                if area>0: any_intersection=True
                if area<=0 or area/(w*h)<r['visibility_threshold']: continue
                norm=[round((l-region[0])/r['width']*1000),round((t-region[1])/r['height']*1000),round((R-region[0])/r['width']*1000),round((b-region[1])/r['height']*1000)]
                expected.append((labels[a['category_id']],norm,a))
            expected.sort(key=lambda o:(o[1][1],o[1][0]))
            text=r['conversations'][1]['value']; parsed=parse_response(text,r['width'],r['height'])
            observed=[(o['label'],o['raw_coordinates']) for o in parsed['objects']]
            if observed!=[(label,box) for label,box,a in expected] or parsed['truncated']: raise ValueError(f'SFT label/coordinate mismatch: {r["id"]}')
            if r['num_objects']!=len(expected) or r['is_negative']!= (not expected): raise ValueError('Target count mismatch')
            if not expected:
                negatives+=1
                if any_intersection or text!='None': raise ValueError('False negative sample')
            if [o['annotation_id'] for o in r['supervision_objects']]!=[a['id'] for label,box,a in sorted(expected,key=lambda o:o[2]['id'])]:
                # IDs are metadata in source annotation order, compare as sets independently of target sorting.
                if {o['annotation_id'] for o in r['supervision_objects']}!={a['id'] for label,box,a in expected}: raise ValueError('Supervision source ID mismatch')
            for label,box,a in expected:
                counts[label]+=1; used_sources.add(a['source_object_id'])
            density['0' if not expected else '1-8' if len(expected)<=8 else '9-32' if len(expected)<=32 else '33-60' if len(expected)<=60 else '>60']+=1
            lengths.append(len(tok.encode(text+'<|im_end|>',add_special_tokens=False)))
        class_sample_counts={label:sum(any(o['label']==label for o in r['supervision_objects']) for r in samples) for label in labels.values()}
        summary[name]={'samples':len(samples),'positive':len(samples)-negatives,'negative':negatives,'physical_boards':len(boards),'targets_with_repeated_crops':sum(r['num_objects'] for r in samples),'class_target_counts':dict(counts),'class_sample_counts':class_sample_counts,'density':dict(density),'answer_tokens_min':min(lengths),'answer_tokens_max':max(lengths),'answer_tokens_median':sorted(lengths)[len(lengths)//2],'over_1024':sum(n>1024 for n in lengths),'over_32768':sum(n>32768 for n in lengths)}
    diff=json.loads((folder/'annotation_diff.json').read_text(encoding='utf-8')); corrected=[d for d in diff if d['status']=='relabeled']
    splits=json.loads((folder/'splits.json').read_text(encoding='utf-8'))['splits']
    eligible=[d for d in corrected if d['image_id'] in splits['train']['image_ids'] + splits['val']['image_ids']]
    test_only=[d for d in corrected if d['image_id'] in splits['test']['image_ids']]
    if not all(d['source_object_id'] in used_sources for d in eligible): raise ValueError('Eligible relabeled connector did not reach SFT')
    if any(d['source_object_id'] in used_sources for d in test_only): raise ValueError('Test connector leaked into SFT')
    return {'passed':True,'splits':summary,'corrected_connectors_in_sft':len(eligible),'corrected_test_connectors_withheld':len(test_only),'test_sft_generated':False,'tokenizer_only_no_model_loaded':True}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--dataset',required=True); ap.add_argument('--sft'); ap.add_argument('--tokenizer'); ap.add_argument('--output',required=True); args=ap.parse_args()
    result={'dataset':audit_dataset(args.dataset)}
    if args.sft: result['sft']=audit_sft(args.dataset,args.sft,args.tokenizer)
    with open(args.output,'x',encoding='utf-8') as stream: json.dump(result,stream,indent=2)
    print(json.dumps(result,indent=2))
if __name__=='__main__': main()
