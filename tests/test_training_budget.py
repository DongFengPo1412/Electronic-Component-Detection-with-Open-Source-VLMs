"""Regression tests for explicit training budgets and masking without loading a VLM."""
import json,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import MagicMock,patch
import torch
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from train_lora import PCBComponentSFTDataset,collate_fn,save_adapter_with_provenance_metadata

class TrainingBudgetTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(); self.root=Path(self.tmp.name); image=self.root/'pic.png'; Image.new('RGB',(28,28),'white').save(image)
  self.rows=[{'id':'small','num_objects':1,'image':str(image),'conversations':[{'value':'<image>\nquestion'},{'value':'answer'}]}, {'id':'dense','num_objects':100,'image':str(image),'conversations':[{'value':'question'},{'value':'a'*4000}]}]
  self.file=self.root/'sft.json'; self.file.write_text(json.dumps(self.rows))
  self.processor=MagicMock(); self.processor.apply_chat_template.return_value='text'; self.processor.tokenizer.encode.return_value=[9]
  self.processor.return_value={'input_ids':torch.tensor([[1,9,2,3]]),'attention_mask':torch.ones(1,4,dtype=torch.long),'pixel_values':torch.zeros(1,1176),'image_grid_thw':torch.tensor([[1,2,2]])}
 def tearDown(self): self.tmp.cleanup()
 def test_no_hidden_filter_when_explicitly_disabled(self):
  self.assertEqual(len(PCBComponentSFTDataset(str(self.file),self.processor,max_objects=None,max_response_len=None)),2)
 def test_legacy_filters_remain_explicit(self): self.assertEqual(len(PCBComponentSFTDataset(str(self.file),self.processor)),1)
 def test_sequence_budget_rejects_instead_of_truncating(self):
  data=PCBComponentSFTDataset(str(self.file),self.processor,max_sequence_length=3)
  with patch('train_lora.process_vision_info',return_value=([],None)):
   with self.assertRaisesRegex(ValueError,'never truncated'): data[0]
 def test_prompt_and_padding_are_not_loss_targets(self):
  data=PCBComponentSFTDataset(str(self.file),self.processor,max_sequence_length=4)
  with patch('train_lora.process_vision_info',return_value=([],None)): b=data[0]
  self.assertEqual(b['labels'].tolist(),[-100,-100,2,3])
  b2={**b,'input_ids':b['input_ids'][:3],'labels':b['labels'][:3],'attention_mask':b['attention_mask'][:3]}; batch=collate_fn([b,b2],pad_token_id=0)
  self.assertEqual(batch['input_ids'][1,-1],0); self.assertEqual(batch['labels'][1,-1],-100); self.assertEqual(batch['attention_mask'][1,-1],0)
 def test_saved_adapter_contains_resume_state(self):
  dest=self.root/'checkpoint'; cfg={'provenance':{'train_sft_path':str(self.file)}}
  save_adapter_with_provenance_metadata(str(dest),MagicMock(),MagicMock(),'local','split',['ArduinoMega'],'sft',5,cfg,{'step':5,'rng_state':torch.get_rng_state()})
  state=torch.load(dest/'trainer_state.pt',weights_only=False); self.assertEqual(state['step'],5); self.assertTrue((dest/'adapter_metadata.json').is_file())

class LastVsBestResumeTest(unittest.TestCase):
 setUp = TrainingBudgetTests.setUp
 tearDown = TrainingBudgetTests.tearDown
 # Use a tiny real torch model and the actual training loop/checkpoint state.
 # Constant validation loss means resume must never promote its final weights.
 def test_resumed_final_is_not_a_new_best(self):
  import contextlib,io,yaml
  from safetensors.torch import save_file,load_file
  import train_lora
  class TinyAdapter(torch.nn.Module):
   def __init__(self):
    super().__init__(); self.base_layer=torch.nn.Linear(2,2,bias=False); self.base_layer.weight.requires_grad_(False); self.lora_B=torch.nn.Linear(2,2,bias=False); self.config=MagicMock()
   def forward(self,**kwargs):
    loss=(self.lora_B.weight-1).square().mean() if self.training else self.lora_B.weight.sum()*0+2
    return type('Output',(),{'loss':loss})()
   def gradient_checkpointing_enable(self): pass
   def print_trainable_parameters(self): pass
   def get_nb_trainable_parameters(self): return (4,8)
   def save_pretrained(self,path): save_file({'lora_B.weight':self.lora_B.weight.detach().cpu().contiguous()},str(Path(path)/'adapter_model.safetensors'))
  def reload(model,path,**kwargs):
   model.lora_B.weight.data.copy_(load_file(str(Path(path)/'adapter_model.safetensors'))['lora_B.weight']); return model
  cfg={'lora':{},'training':{'learning_rate':0.001,'gradient_accumulation_steps':1,'save_steps':1},'data':{'max_length':4}}
  conf=self.root/'config.yaml'; conf.write_text(yaml.safe_dump(cfg)); split=self.root/'splits.json'; split.write_text('{}')
  self.processor.tokenizer.pad_token_id=0
  first=self.root/'first'; second=self.root/'second'
  common=['train_lora','--model-path','toy','--data-file',str(self.file),'--val-data-file',str(self.file),'--splits-path',str(split),'--config',str(conf),'--max-steps','2','--eval-steps','1','--device','cpu']
  patches=[patch.object(train_lora.Qwen2VLForConditionalGeneration,'from_pretrained',side_effect=lambda *a,**k:TinyAdapter()),patch.object(train_lora.AutoProcessor,'from_pretrained',return_value=self.processor),patch.object(train_lora,'get_peft_model',side_effect=lambda m,c:m),patch.object(train_lora.PeftModel,'from_pretrained',side_effect=reload),patch('peft.get_peft_model_state_dict',side_effect=lambda m:{'lora_B.weight':m.lora_B.weight}),patch.object(train_lora,'validate_samples',return_value=['toy']),patch.object(train_lora,'model_identity',return_value='toy-content'),patch.object(train_lora,'source_identity',return_value='toy-source'),patch.object(train_lora,'code_identity',return_value='toy-commit'),patch.object(train_lora,'dependencies',return_value={}),patch.object(train_lora,'verify_adapter',side_effect=lambda path,*a:json.loads((Path(path)/'adapter_metadata.json').read_text())),patch.object(train_lora,'process_vision_info',return_value=([],None)),patch('torch.cuda.is_available',return_value=False)]
  with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
   for item in patches: stack.enter_context(item)
   with patch.object(sys,'argv',common+['--output-dir',str(first),'--stop-after-step','1']): train_lora.main()
   with patch.object(sys,'argv',common+['--output-dir',str(second),'--resume-from-checkpoint',str(first/'checkpoint-1')]): train_lora.main()
  self.assertTrue((first/'best_adapter/adapter_model.safetensors').is_file())
  self.assertFalse((second/'best_adapter').exists())
  self.assertTrue((second/'final_adapter/adapter_model.safetensors').is_file())
  state=torch.load(second/'final_adapter/trainer_state.pt',weights_only=False)
  self.assertEqual(state['step'],2); self.assertEqual(state['scheduler_state_dict']['last_epoch'],2)
  self.assertEqual(state['best_adapter_source'],str((first/'best_adapter').resolve()))
 def test_resume_at_target_step_saves_final_without_unbound_error(self):
  import contextlib,io,yaml
  from safetensors.torch import save_file,load_file
  import train_lora
  class TinyAdapter(torch.nn.Module):
   def __init__(self):
    super().__init__(); self.base_layer=torch.nn.Linear(2,2,bias=False); self.base_layer.weight.requires_grad_(False); self.lora_B=torch.nn.Linear(2,2,bias=False); self.config=MagicMock()
   def forward(self,**kwargs):
    return type('Output',(),{'loss':(self.lora_B.weight-1).square().mean()})()
   def gradient_checkpointing_enable(self): pass
   def print_trainable_parameters(self): pass
   def get_nb_trainable_parameters(self): return (4,8)
   def save_pretrained(self,path): save_file({'lora_B.weight':self.lora_B.weight.detach().cpu().contiguous()},str(Path(path)/'adapter_model.safetensors'))
  def reload(model,path,**kwargs):
   model.lora_B.weight.data.copy_(load_file(str(Path(path)/'adapter_model.safetensors'))['lora_B.weight']); return model
  cfg={'lora':{},'training':{'learning_rate':0.001,'gradient_accumulation_steps':1,'save_steps':1},'data':{'max_length':4}}
  conf=self.root/'config.yaml'; conf.write_text(yaml.safe_dump(cfg)); split=self.root/'splits.json'; split.write_text('{}')
  self.processor.tokenizer.pad_token_id=0
  first=self.root/'first_zero'; second=self.root/'second_zero'
  common=['train_lora','--model-path','toy','--data-file',str(self.file),'--val-data-file',str(self.file),'--splits-path',str(split),'--config',str(conf),'--max-steps','1','--eval-steps','1','--device','cpu']
  patches=[patch.object(train_lora.Qwen2VLForConditionalGeneration,'from_pretrained',side_effect=lambda *a,**k:TinyAdapter()),patch.object(train_lora.AutoProcessor,'from_pretrained',return_value=self.processor),patch.object(train_lora,'get_peft_model',side_effect=lambda m,c:m),patch.object(train_lora.PeftModel,'from_pretrained',side_effect=reload),patch('peft.get_peft_model_state_dict',side_effect=lambda m:{'lora_B.weight':m.lora_B.weight}),patch.object(train_lora,'validate_samples',return_value=['toy']),patch.object(train_lora,'model_identity',return_value='toy-content'),patch.object(train_lora,'source_identity',return_value='toy-source'),patch.object(train_lora,'code_identity',return_value='toy-commit'),patch.object(train_lora,'dependencies',return_value={}),patch.object(train_lora,'verify_adapter',side_effect=lambda path,*a:json.loads((Path(path)/'adapter_metadata.json').read_text())),patch.object(train_lora,'process_vision_info',return_value=([],None)),patch('torch.cuda.is_available',return_value=False)]
  with contextlib.ExitStack() as stack,contextlib.redirect_stdout(io.StringIO()):
   for item in patches: stack.enter_context(item)
   with patch.object(sys,'argv',common+['--output-dir',str(first)]): train_lora.main()
   with patch.object(sys,'argv',common+['--output-dir',str(second),'--resume-from-checkpoint',str(first/'checkpoint-1')]): train_lora.main()
  self.assertTrue((second/'final_adapter/trainer_state.pt').is_file())
  state=torch.load(second/'final_adapter/trainer_state.pt',weights_only=False)
  self.assertEqual(state['step'],1)

if __name__=='__main__': unittest.main()
