"""P0 regression tests: small synthetic inputs, no model downloads or GPU work."""
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parents[1] / 'scripts')]
from pcb_vlm import provenance as pv
from pcb_vlm.config import validate_category_consistency, get_canonical_categories
from evaluate import evaluate_detections, compute_official_coco_eval, is_valid_box
from infer_batch import get_adapter_hash, compute_prediction_cache_key, run_batch_inference

class P0Tests(unittest.TestCase):
    def setUp(self):
        self.gt={'images':[{'id':1,'width':20,'height':20}], 'categories':[{'id':1,'name':'ic'}],
                 'annotations':[{'id':1,'image_id':1,'category_id':1,'bbox':[1,1,4,4],'area':16,'iscrowd':0}]}
    def test_extra_and_duplicate_classes_rejected(self):
        for cats in [get_canonical_categories()+['unknown'], get_canonical_categories()+['ic']]:
            with self.assertRaises(ValueError): validate_category_consistency(cats)
    def test_strict_bounds(self):
        self.assertFalse(is_valid_box([0,0,20.01,20],20,20))
    def test_duplicate_boxes_are_false_positives(self):
        obj={'label':'ic','bbox_xyxy':[1,1,5,5]}
        m=evaluate_detections(self.gt,{'predictions':[{'image_id':1,'status':'success','objects':[obj,obj]}]})
        self.assertEqual((m['tp'],m['fp'],m['fn']),(1,1,0))
    def test_failure_and_missing_count_as_fn(self):
        for records in [[],[{'image_id':1,'status':'inference_failed','objects':[]}]]:
            m=evaluate_detections(self.gt,{'predictions':records}); self.assertEqual(m['fn'],1)
            self.assertEqual(m['effective_parse_rate'],0)
    def test_unknown_category_and_nan_score(self):
        for label,score in [('nonsense',1),('ic',float('nan'))]:
            m=evaluate_detections(self.gt,{'predictions':[{'image_id':1,'status':'success','objects':[{'label':label,'score':score,'bbox_xyxy':[1,1,5,5]}]}]})
            self.assertEqual(m['tp'],0)
    def test_missing_adapter_rejected(self):
        with self.assertRaises(FileNotFoundError): get_adapter_hash('does-not-exist-p0')
        with self.assertRaises(FileNotFoundError): pv.verify_adapter('does-not-exist-p0','unused')
    def test_cache_changes_with_generation_model_and_nms(self):
        base=compute_prediction_cache_key('image','model','prompt',None,None)
        for extra in [dict(max_new_tokens=256),dict(repetition_penalty=1.1),dict(model_sha256='changed'),dict(nms_iou=.3),dict(do_sample=True)]:
            self.assertNotEqual(base,compute_prediction_cache_key('image','model','prompt',None,None,**extra))
    def test_unknown_physical_id_rejected(self):
        with self.assertRaises(ValueError): pv.physical_id('unmapped_board')
    def test_incomplete_sft_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); parts={}; manifest=[]
            for i,name in enumerate(('train','val','test'),1):
                folder=f'folder{i}'; manifest.append({'image_id':i,'folder':folder})
                parts[name]={'image_ids':[i],'image_folders':[folder],'boards':[f'board{i}']}
            (d/'splits.json').write_text(json.dumps({'splits':parts}))
            (d/'images_manifest.json').write_text(json.dumps(manifest))
            with patch.object(pv,'canonical_map',return_value={f'folder{i}':f'board{i}' for i in range(1,4)}):
                with self.assertRaisesRegex(ValueError,'Incomplete'):
                    pv.validate_samples([{'id':'legacy'}],'train',d/'splits.json')
                parts['val']['boards']=['board1']
                with self.assertRaises(ValueError): pv.validate_partition({'splits':parts},manifest)
    def test_coco_limits_apply_and_are_separate(self):
        gt=copy.deepcopy(self.gt); gt['images'][0].update(width=1000,height=20)
        gt['annotations']=[dict(id=i+1,image_id=1,category_id=1,bbox=[i*6,1,4,4],area=16,iscrowd=0) for i in range(110)]
        objs=[{'label':'ic','score':1.,'bbox_xyxy':[i*6,1,i*6+4,5]} for i in range(110)]
        m=compute_official_coco_eval(gt,{'predictions':[{'image_id':1,'status':'success','objects':objs}]})
        self.assertEqual(m['maxDets'],[1,10,100]); self.assertEqual(m['dense_extended']['maxDets'],[1,10,1000])
        self.assertGreater(m['dense_extended']['AP_50'],m['AP_50'])
    def test_protocol_test_split_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            import yaml
            f=Path(d)/'bad.yaml'; f.write_text(yaml.safe_dump({'dataset_canonical_dir':d,'test_split':{'image_ids':[999],'boards':[]},'categories':get_canonical_categories()}))
            with patch.object(pv,'read_json',return_value={'splits':{'test':{'image_ids':[1],'boards':[]}}}),patch.object(pv,'validate_partition'):
                with self.assertRaisesRegex(ValueError,'Frozen test split'): pv.load_protocol(f)
    def test_enriched_protocol_tampered_hash_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            import yaml
            d = Path(d)
            (d / 'splits.json').write_text('{}')
            (d / 'coco_annotations.json').write_text('{}')
            (d / 'images_manifest.json').write_text('[]')
            cats = [{'id': i + 1, 'name': n, 'chinese_name': n} for i, n in enumerate(get_canonical_categories())]
            proto = {
                'dataset_canonical_dir': str(d),
                'test_split': {'image_ids': [1], 'boards': ['b1']},
                'categories': get_canonical_categories(),
                'generation': {'max_pixels': 602112, 'repetition_penalty': 1.25},
                'annotations_sha256': pv.sha256(d / 'coco_annotations.json'),
                'split_sha256': pv.sha256(d / 'splits.json'),
                'final_protocol_hash': '0' * 64,
            }
            f = d / 'tampered.yaml'
            f.write_text(yaml.safe_dump(proto))
            with patch.object(pv, 'read_json', side_effect=[{'splits': {'test': {'image_ids': [1], 'boards': ['b1']}}}, [], {'categories': cats}]), \
                 patch.object(pv, 'validate_partition'), \
                 patch('pcb_vlm.config.get_category_definitions', return_value=cats):
                with self.assertRaisesRegex(ValueError, 'final_protocol_hash mismatch'):
                    pv.load_protocol(f)
    def test_failed_inference_is_not_completed(self):
        # Inspect the real output path through a lightweight mocked adapter, no VLM weights loaded.
        with tempfile.TemporaryDirectory() as d:
            d=Path(d); image=d/'image'; image.write_bytes(b'picture')
            splits=d/'splits.json'; splits.write_text(json.dumps({'splits':{'val':{'image_ids':[1]}}}))
            manifest=d/'manifest.json'; manifest.write_text(json.dumps([{'image_id':1,'image_path':str(image),'sha256':pv.sha256(image),'width':20,'height':20,'folder':'x'}]))
            protocol={'split_sha256':pv.sha256(splits),'models':{'qwen_base':'local'},'generation':{'max_pixels':602112,'max_new_tokens':16,'repetition_penalty':1.25,'do_sample':False},'tiling':{'nms_iou_threshold':.5},'prompt':'test'}
            with patch('infer_batch.load_protocol',return_value=protocol),patch('infer_batch.model_identity',return_value='pinned'),patch('infer_batch.QwenAdapter') as adapter:
                adapter.return_value.requested_model='local'; adapter.return_value.model_path='local'
                adapter.return_value.predict_image.side_effect=RuntimeError('synthetic inference failure')
                result=run_batch_inference(str(manifest),str(splits),'val','local',str(d/'out'),'smoke',device='cpu')
                self.assertEqual(result['execution_status'],'partial_failed')
                self.assertEqual(result['status_counts'],{'inference_failed':1})

if __name__=='__main__': unittest.main()
