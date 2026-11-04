"""P1 strict coordinate contracts, adapter state and tile mapping regressions."""
import sys
import unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from pcb_vlm.geometry.coordinate_protocol import parse_response,to_pixels,generation_finished
from pcb_vlm.models.qwen_adapter import QwenAdapter
from pcb_vlm.models.deepseek_adapter import DeepSeekAdapter
from pcb_vlm.geometry.tiling import map_tile_box_to_global

class ExplicitProtocolTests(unittest.TestCase):
 def test_all_units_and_orders(self):
  for name,values in [('xyxy_norm1000',[100,200,400,600]),('xyxy_norm1',[.1,.2,.4,.6]),('xyxy_pixels',[80,80,320,240]),('yxyx_norm1000',[200,100,600,400]),('yxyx_norm1',[.2,.1,.6,.4]),('yxyx_pixels',[80,80,240,320])]:
   with self.subTest(name=name): self.assertEqual(to_pixels(values,800,400,name),[80,80,320,240])
 def test_near_origin_never_auto_rescaled(self):
  self.assertEqual(to_pixels([0,0,1,1],800,400,'xyxy_norm1000'),[0,0,.8,.4])
  self.assertEqual(to_pixels([0,0,1,1],800,400,'xyxy_norm1'),[0,0,800,400])
 def test_boundaries_and_shapes(self):
  for w,h in [(512,512),(800,400),(400,800)]:
   self.assertEqual(to_pixels([0,0,1000,1000],w,h),[0,0,w,h])
 def test_reversed_negative_outside_and_nan_rejected(self):
  for box in [[300,100,200,200],[-1,0,200,300],[0,0,1001,500],[0,0,float('nan'),5],[0,0,3,float('inf')]]:
   with self.assertRaises(ValueError): to_pixels(box,800,400)
 def test_unknown_protocol_rejected(self):
  with self.assertRaises(ValueError): to_pixels([1,2,3,4],800,400,'auto')
 def test_thin_and_small_boxes_are_retained(self):
  r=parse_response('resistor [0,0,100,1]',800,400)
  self.assertEqual(len(r['objects']),1)
 def test_native_qwen_and_mixed_text(self):
  s='Here: <|object_ref_start|>ic<|object_ref_end|><|box_start|>(100,200),(400,600)<|box_end|> and capacitor [500,200,600,300]'
  r=parse_response(s,800,400); self.assertEqual(r['status'],'success'); self.assertEqual(len(r['objects']),2)
 def test_truncated_multiobject(self):
  r=parse_response('ic [100,200,400,600], resistor <|box_start|>(500,',800,400)
  self.assertTrue(r['truncated']); self.assertEqual(r['status'],'partial_success'); self.assertEqual(len(r['objects']),1)
 def test_duplicates_preserved_and_counted(self):
  r=parse_response('ic [100,200,400,600], ic [100,200,400,600]',800,400)
  self.assertEqual(len(r['objects']),2); self.assertEqual(r['audit']['duplicate_boxes'],1)
 def test_unknown_or_missing_labels_are_not_guessed(self):
  for s in ['weird [100,200,400,600]','[100,200,400,600]','<|box_start|>(100,200),(400,600)<|box_end|>']:
   r=parse_response(s,800,400); self.assertEqual(r['objects'],[]); self.assertGreater(r['audit']['unknown_categories'],0)
 def test_empty_prose_none_and_special_tokens(self):
  for s,status in [('', 'parse_failed'),('none of these resistors can be identified','parse_failed'),('None<|im_end|>','no_objects_detected'),('None</s>','no_objects_detected')]:
   self.assertEqual(parse_response(s,800,400)['status'],status)
 def test_vl2_not_native_for_deepseek_v1(self):
  a=object.__new__(DeepSeekAdapter)
  r=a.parse_grounding_output_with_status('<|ref|>ic<|/ref|><|det|>[[100,200,400,600]]<|/det|>',800,400)
  self.assertEqual(r[0],[]); self.assertEqual(r[1],'parse_failed'); self.assertIn('VL2',r[2])
 def test_deepseek_project_contract_and_axis_distinction(self):
  a=object.__new__(DeepSeekAdapter); r=a.parse_grounding_output('ic [200,100,600,400]',800,400)
  self.assertEqual(r[0]['bbox_xyxy'],[80,80,320,240])
  self.assertNotEqual(r[0]['bbox_xyxy'],to_pixels([200,100,600,400],800,400))
 def test_legacy_is_explicit_and_marked(self):
  a=object.__new__(QwenAdapter); self.assertEqual(a.parse_grounding_output('[.1,.2,.4,.6]',800,400),[])
  a.parse_grounding_output('[0.1,0.2,0.4,0.6]',800,400,legacy=True)
  self.assertFalse(a.last_parse_audit['formal_eligible'])
 def test_actual_tile_mapping(self):
  box=to_pixels([100,200,400,600],512,512)
  result=map_tile_box_to_global(box,100,200,1000,1000)
  self.assertEqual(result,[151.2,302.4,304.8,507.2])
 def test_eos_is_token_based(self):
  self.assertTrue(generation_finished([42,2],2)); self.assertFalse(generation_finished([42,3],2))
 def test_formal_adapters_record_same_contract(self):
  a=object.__new__(QwenAdapter); boxes,status,detail,trunc=a.parse_grounding_output_with_status('ic [100,200,400,600]',800,400)
  self.assertEqual(a.last_parse_audit['final_boxes'],1); self.assertEqual(boxes[0]['coordinate_protocol'],'xyxy_norm1000')

class AdditionalContracts(unittest.TestCase):
 def test_bbox_first_dictionary_keeps_label(self):
  r=parse_response('{"bbox":[100,200,400,600], "label":"ic"}',800,400)
  self.assertEqual(r['objects'][0]['label'],'ic')
 def test_wrong_native_family_rejected(self):
  r=parse_response('ic <|box_start|>(100,200),(400,600)<|box_end|>',800,400,'yxyx_norm1000',False)
  self.assertEqual(r['status'],'parse_failed')
 def test_geometry_helpers_reject_invalid_axis_or_box(self):
  from pcb_vlm.geometry.box_ops import qwen_norm_to_xyxy,xyxy_to_qwen_norm
  with self.assertRaises(ValueError): qwen_norm_to_xyxy([0,0,100,100],800,400,'guess')
  with self.assertRaises(ValueError): xyxy_to_qwen_norm([-1,0,100,100],800,400)
 def test_project_protocol_is_in_frozen_manifest(self):
  from pcb_vlm.provenance import load_protocol
  p=load_protocol()
  self.assertEqual(p['coordinate_protocols']['qwen'],'xyxy_norm1000')
  self.assertEqual(p['coordinate_protocols']['deepseek'],'yxyx_norm1000')

if __name__=='__main__': unittest.main()
