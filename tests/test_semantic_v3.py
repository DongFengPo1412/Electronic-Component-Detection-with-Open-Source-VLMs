"""V3 full-source reconstruction, provenance and reconciliation regression tests."""
import copy,json,sys,tempfile,unittest,xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch
from PIL import Image
import yaml
ROOT=Path(__file__).resolve().parents[1]; sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from pcb_vlm.data.raw_labels import classify_raw_name
from pcb_vlm.provenance import sha256,stamp_samples,validate_samples
from rebuild_semantic_dataset import rebuild
from audit_semantic_v3 import audit_dataset
from build_sft_data import process_split_sft
POLICY=yaml.safe_load((ROOT/'configs/semantic_policy_v3.yaml').read_text(encoding='utf-8-sig'))

class SemanticV3Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name)
 def tearDown(self): self.tmp.cleanup()
 def fixture(self):
  raw=self.root/'raw'; old=self.root/'old'; old.mkdir(); rows=[]; images=[]; anns=[]; parts={}
  objects=[ [('text "LED1"',[0,0,10,10],5),('connector ICSP',[20,0,30,10],3),('capacitor UNKNOWN',[40,0,50,10],None),('resistor R1',[60,0,70,10],2),('unknown UNLABELED',[70,20,80,30],5),('"resistor jumper" RJ1',[80,0,90,10],2)], [('ic U1',[1,1,10,10],3)], [('led LED2',[1,1,10,10],5)] ]
  for i,(folder,board,split) in enumerate([('ArduinoMega_Top','ArduinoMega','train'),('Arty_Top','Arty','val'),('HackRF','HackRF','test')],1):
   d=raw/folder; d.mkdir(parents=True); pic=d/(folder+'.png'); Image.new('RGB',(100,100),(i*30,90,120)).save(pic)
   entry=dict(image_id=i,folder=folder,board_group_id=board,width=100,height=100,sha256=sha256(pic),image_path=str(pic),file_name=pic.name); rows.append(entry); images.append({'id':i,**entry}); root=ET.Element('annotation')
   for name,box,cid in objects[i-1]:
    o=ET.SubElement(root,'object'); ET.SubElement(o,'name').text=name; b=ET.SubElement(o,'bndbox')
    for k,v in zip(('xmin','ymin','xmax','ymax'),box): ET.SubElement(b,k).text=str(v)
    if cid is not None:
     x,y,X,Y=box; anns.append(dict(id=len(anns)+1,image_id=i,category_id=cid,bbox=[x,y,X-x,Y-y],bbox_xyxy=box,area=(X-x)*(Y-y),iscrowd=0,board_group_id=board))
   ET.ElementTree(root).write(d/(folder+'.xml'))
   parts[split]=dict(boards=[board],image_ids=[i],image_folders=[folder],num_annotations=sum(a['image_id']==i for a in anns))
  categories=yaml.safe_load((ROOT/'configs/classes.yaml').read_text(encoding='utf-8'))['categories']
  for filename,value in [('coco_annotations.json',{'info':{'version':'old'},'images':images,'annotations':anns,'categories':categories}),('images_manifest.json',rows),('splits.json',{'metadata':{},'splits':parts})]: (old/filename).write_text(json.dumps(value),encoding='utf-8')
  new=self.root/'new'; summary=rebuild(old,raw,new,ROOT/'configs/semantic_policy_v3.yaml'); return old,new,summary
 def test_casefold_and_compound_class(self):
  r=classify_raw_name('  "Resistor Network" UNKNOWN',POLICY); self.assertEqual((r['decision'],r['category']),('keep','resistor'))
 def test_designator_never_changes_class(self): self.assertEqual(classify_raw_name('connector ICSP',POLICY)['category'],'connector')
 def test_unknown_class_and_unknown_value_differ(self):
  self.assertEqual(classify_raw_name('unknown LED',POLICY)['decision'],'ignore'); self.assertEqual(classify_raw_name('capacitor UNKNOWN',POLICY)['decision'],'keep')
 def test_unrecognized_or_malformed_require_review(self):
  for text in [None,'','"text LED','not-a-real-class R1']:
   self.assertEqual(classify_raw_name(text,POLICY)['decision'],'ambiguous')
 def test_rebuild_diff_includes_added_and_relabel(self):
  old,new,s=self.fixture(); self.assertEqual(s['diff_counts'],{'removed':3,'relabeled':1,'unchanged':3,'added':1}); self.assertEqual(s['new_annotations'],5)
 def test_old_omitted_target_is_restored(self):
  old,new,s=self.fixture(); ledger=json.loads((new/'raw_objects.json').read_text(encoding='utf-8')); self.assertEqual(next(r for r in ledger if r['raw_name']=='capacitor UNKNOWN')['decision'],'keep'); self.assertTrue(audit_dataset(new)['passed'])
 def test_split_identity_preserved(self):
  old,new,s=self.fixture(); a=json.loads((old/'splits.json').read_text(encoding='utf-8')); b=json.loads((new/'splits.json').read_text(encoding='utf-8'))
  for split in ('train','val','test'):
   for key in ('boards','image_ids','image_folders'): self.assertEqual(a['splits'][split][key],b['splits'][split][key])
 def test_raw_coverage_catches_missing_target(self):
  old,new,s=self.fixture(); f=new/'coco_annotations.json'; g=json.loads(f.read_text(encoding='utf-8')); g['annotations'].pop(); f.write_text(json.dumps(g))
  with self.assertRaises((KeyError,ValueError)): audit_dataset(new)
 def test_preserve_old_and_reject_output_reuse(self):
  old,new,s=self.fixture(); before=sha256(old/'coco_annotations.json')
  with self.assertRaises(FileExistsError): rebuild(old,self.root/'raw',new,ROOT/'configs/semantic_policy_v3.yaml')
  self.assertEqual(sha256(old/'coco_annotations.json'),before)
 def test_sft_versions_hashes_and_connector_propagate(self):
  old,new,s=self.fixture(); g=json.loads((new/'coco_annotations.json').read_text(encoding='utf-8')); man=json.loads((new/'images_manifest.json').read_text(encoding='utf-8')); out=self.root/'sft'; out.mkdir(); byimg={1:[a for a in g['annotations'] if a['image_id']==1]}; labels={c['id']:c['name'] for c in g['categories']}
  rows=process_split_sft('train',[1],man,byimg,labels,list(labels.values()),str(out),str(out/'crops'),1024,.2,.5); self.assertEqual(len(rows),1)
  stamp_samples(rows,'train',new/'splits.json',man); validate_samples(rows,'train',new/'splits.json'); self.assertIn('connector',rows[0]['conversations'][1]['value']); self.assertEqual(rows[0]['num_objects'],3)
  rows[0]['annotations_sha256']='fake'
  with self.assertRaisesRegex(ValueError,'annotation/category'): validate_samples(rows,'train',new/'splits.json')

if __name__=='__main__': unittest.main()
