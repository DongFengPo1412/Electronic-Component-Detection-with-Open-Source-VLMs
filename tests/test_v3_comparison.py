import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from run_v3_lora_comparison import crop_ground_truth,diagnostic_tile,detection_rank,restore_global
from evaluate import evaluate_detections
class FrozenComparisonTests(unittest.TestCase):
 def test_crop_keeps_all_intersections_not_just_sft_targets(self):
  gt={'images':[{'id':1,'width':100,'height':80}], 'categories':[{'id':1,'name':'ic'}], 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[5,5,30,20]},{'id':2,'image_id':1,'category_id':1,'bbox':[25,20,5,5]}]}
  crop=crop_ground_truth(gt,1,[20,10,50,40]); self.assertEqual(len(crop['annotations']),2);self.assertEqual(crop['annotations'][0]['bbox'],[0,0,15,15]); self.assertEqual(evaluate_detections(crop,[],expected_image_ids=[1])['fn'],2)
 def test_tile_selection_independent_of_labels(self):
  a={'width':4112,'height':3008}; first=diagnostic_tile(a); self.assertEqual(first,diagnostic_tile({**a,'ignored_labels':['resistor']}));self.assertEqual(len(first['bbox']),4)
 def test_selection_prefers_detection_f1_then_earlier_step(self):
  a={'tp':3,'fp':20,'fn':10}; b={'tp':2,'fp':0,'fn':11}; self.assertGreater(detection_rank(b,80),detection_rank(a,40));self.assertGreater(detection_rank(b,40),detection_rank(b,80))
 def test_merge_offsets_preserves_failed_tile_status(self):
  row={'tile_bbox':[10,20,30,40],'objects':[{'label':'ic','bbox_xyxy':[1,2,5,7]}],'status':'partial_success','is_truncated':True}
  result=restore_global([row],{'image_id':1,'width':100,'height':100});self.assertEqual(result['objects'][0]['bbox_xyxy'],[11,22,15,27]);self.assertEqual(result['status'],'partial_success');self.assertTrue(result['is_truncated'])
if __name__=='__main__':unittest.main()
