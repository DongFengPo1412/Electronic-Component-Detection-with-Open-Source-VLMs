import copy,json,sys,tempfile,unittest
from pathlib import Path
from PIL import Image
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
from pcb_vlm.data.yolo_v3 import category_mapping,xyxy_to_yolo,yolo_to_xyxy,clip_objects,source_regions,build_dataset,audit_dataset,read,write
from pcb_vlm.provenance import canonical_map,sha256
from pcb_vlm.geometry.tiling import map_tile_box_to_global,nms_merge_predictions
from infer_yolo_v3 import yolo_objects,confidence_subset,evaluate_prediction_files
from evaluate import evaluate_detections
class YoloV3Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.source=self.root/'source';self.source.mkdir();mapping=canonical_map();folders=[];seen=set()
  for f,b in mapping.items():
   if b not in seen:folders.append((f,b));seen.add(b)
   if len(folders)==3:break
  manifest=[];parts={};annotations=[]
  for i,(part,(folder,board)) in enumerate(zip(('train','val','test'),folders),1):
   file=self.source/f'{i}.png';Image.new('RGB',(100,60),(i*30,40,50)).save(file);manifest.append({'image_id':i,'folder':folder,'physical_board_id':board,'width':100,'height':60,'image_path':str(file),'sha256':sha256(file)});parts[part]={'image_ids':[i],'image_folders':[folder],'boards':[board]};annotations.extend([{'id':i*10,'image_id':i,'category_id':17,'bbox':[10,5,1,1]},{'id':i*10+1,'image_id':i,'category_id':42,'bbox':[70,3,29,55]}])
  self.gt={'info':{'version':'3.0-semantic'},'images':[{'id':r['image_id'],'width':100,'height':60} for r in manifest],'categories':[{'id':42,'name':'resistor'},{'id':17,'name':'capacitor'}],'annotations':annotations};write(self.source/'coco_annotations.json',self.gt);write(self.source/'images_manifest.json',manifest);write(self.source/'splits.json',{'splits':parts})
 def tearDown(self):self.tmp.cleanup()
 def built(self):
  out=self.root/'generated';data=build_dataset(self.source,out,size=40,overlap=.2);return out,data
 def test_completed_epoch_uses_csv_not_final_validation_callback(self):
  from train_yolo_v3 import completed_epochs
  f=self.root/'results.csv';f.write_text('epoch,loss\n1,0.5\n2,0.4\n',encoding='utf-8');self.assertEqual(completed_epochs(f),2)
 def test_sparse_category_id_mapping(self):self.assertEqual(category_mapping(self.gt['categories']),({17:0,42:1},['capacitor','resistor']))
 def test_non_square_coordinates(self):
  b=[10,5,90,55];self.assertEqual(xyxy_to_yolo(b,100,60),[.5,.5,.8,50/60]);self.assertTrue(all(abs(a-b)<1e-9 for a,b in zip(yolo_to_xyxy(xyxy_to_yolo(b,100,60),100,60),b)))
 def test_subpixel_targets_not_rounded_away(self):
  b=[1,1,1.001,1.002];norm=xyxy_to_yolo(b,640,640);self.assertGreater(float(f'{norm[2]:.12f}'),0);self.assertGreater(float(f'{norm[3]:.12f}'),0)
 def test_edge_clipping_keeps_visible_part(self):
  a=[{'id':1,'category_id':17,'bbox':[35,10,10,10]}];r=clip_objects(a,[0,0,40,40],{17:0});self.assertEqual(r[0]['local_xyxy'],[35,10,40,20]);self.assertEqual(r[0]['visibility'],.5)
 def test_any_target_intersection_is_not_background(self):self.assertEqual(len(clip_objects([{'id':1,'category_id':17,'bbox':[39.9,0,4,4]}],[0,0,40,40],{17:0})),1)
 def test_full_context_for_large_targets(self):
  r=source_regions({'width':100,'height':60},self.gt['annotations'][:2],{17:0,42:1},40,.2);self.assertIn((0,0,100,60),r)
 def test_build_and_independent_audit(self):
  out,data=self.built();r=audit_dataset(out);self.assertTrue(r['passed']);self.assertEqual(r['summary']['train']['unique_gt'],2);self.assertEqual(r['summary']['train']['fully_visible_unique_gt'],2)
 def test_test_data_never_generated(self):
  out,data=self.built();self.assertEqual({r['split'] for r in data['records']},{'train','val'});self.assertNotIn(3,{r['image_id'] for r in data['records']})
 def test_physical_board_conflict_rejected(self):
  out,data=self.built();data['records'][0]['physical_board_id']='wrong';write(out/'provenance.json',data)
  with self.assertRaises(AssertionError):audit_dataset(out)
 def test_duplicate_source_region_rejected(self):
  out,data=self.built();data['records'].append(copy.deepcopy(data['records'][0]));write(out/'provenance.json',data)
  with self.assertRaises(AssertionError):audit_dataset(out)
 def test_small_target_loader_label_precision(self):
  out,data=self.built();self.assertGreater(data['summary']['train']['under2px_occurrences'],0);self.assertTrue(audit_dataset(out)['passed'])
 def test_global_coordinate_restore(self):self.assertEqual(map_tile_box_to_global([1,2,5,7],10,20,100,60),[11.,22.,15.,27.])
 def test_nms_keeps_real_confidence(self):
  rows=[{'label':'capacitor','bbox_xyxy':[0,0,5,5],'score':p} for p in [.2,.8]];kept=nms_merge_predictions(rows,.5);self.assertEqual(len(kept),1);self.assertEqual(kept[0]['score'],.8)
 def test_output_not_capped_at_300_and_duplicates_penalized(self):
  objects=yolo_objects([[0,0,5,5]]*501,[0]*501,[.8]*501,['capacitor'],100,60);self.assertEqual(len(objects),501);gt={'images':[{'id':1,'width':100,'height':60}],'categories':[{'id':1,'name':'capacitor'}],'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[0,0,5,5]}]};m=evaluate_detections(gt,[{'image_id':1,'status':'success','objects':objects}]);self.assertEqual((m['tp'],m['fp']),(1,500))
 def test_score_and_fixed_threshold_preserved(self):
  objects=yolo_objects([[0,0,5,5]],[0],[.712345],['capacitor'],100,60);self.assertEqual(objects[0]['score'],.712345);self.assertEqual(len(confidence_subset([{'objects':objects}],.8)[0]['objects']),0)
 def test_missing_prediction_retains_denominator_and_format(self):
  m=evaluate_detections(self.gt,[]);self.assertEqual(m['total_gt'],6);self.assertEqual(m['fn'],6)
  obj=yolo_objects([[10,5,11,6]],[0],[.9],['capacitor'],100,60);m=evaluate_detections(self.gt,[{'image_id':1,'status':'success','objects':obj}]);self.assertEqual(m['tp'],1);self.assertEqual(m['fn'],5)
 def test_unknown_class_or_nonfinite_score_rejected(self):
  with self.assertRaises(ValueError):yolo_objects([[0,0,1,1]],[9],[.5],['capacitor'],100,60)
  with self.assertRaises(ValueError):yolo_objects([[0,0,1,1]],[0],[float('nan')],['capacitor'],100,60)
if __name__=='__main__':unittest.main()
