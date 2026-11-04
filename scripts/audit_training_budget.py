"""CPU-only coverage and real processor/collator sequence audit; never truncates labels."""
import argparse,collections,csv,json,sys,time
from pathlib import Path
sys.path[:0]=[str(Path(__file__).resolve().parents[1]/'src'),str(Path(__file__).resolve().parents[1]/'scripts')]
import torch
from transformers import AutoProcessor
from train_lora import PCBComponentSFTDataset,collate_fn

def coverage(records):
    occurrences=collections.Counter(); unique=collections.defaultdict(set); class_samples=collections.Counter(); boards=set()
    for r in records:
        boards.add(r['physical_board_id']); seen=set()
        for o in r['supervision_objects']:
            occurrences[o['label']]+=1; unique[o['label']].add(o['annotation_id']); seen.add(o['label'])
        class_samples.update(seen)
    return {'records':len(records),'target_occurrences':sum(occurrences.values()),'unique_annotations':len(set().union(*unique.values())) if unique else 0,'physical_boards':len(boards),'negative_records':sum(r['is_negative'] for r in records),'high_density_over32':sum(r['num_objects']>32 for r in records),'class_occurrences':dict(occurrences),'class_unique_annotations':{k:len(v) for k,v in unique.items()},'class_records':dict(class_samples),'board_ids':sorted(boards)}

def audit(sft_dir,model_dir,output,pixels=200704,sequence_budget=1536):
    torch.set_num_threads(2); output=Path(output); output.mkdir(parents=True,exist_ok=True); processor=AutoProcessor.from_pretrained(model_dir,local_files_only=True); all_summary={}; all_rows=[]
    for split in ('train','val'):
        path=Path(sft_dir)/f'{split}_sft.json'; records=json.loads(path.read_text(encoding='utf-8')); data=PCBComponentSFTDataset(str(path),processor,max_pixels=pixels,max_objects=None,max_response_len=None); kept=[]; legacy_kept=[]; rows=[]; saved=[]
        for i,r in enumerate(records):
            b=data[i]; target=b['labels'][b['labels']!=-100]; decoded=processor.tokenizer.decode(target,skip_special_tokens=False)
            if not decoded.startswith(r['conversations'][1]['value']) or '<|im_end|>' not in decoded: raise ValueError('Incomplete assistant target/mask')
            image_id=processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
            if ((b['input_ids']==image_id)&(b['labels']!=-100)).any(): raise ValueError('Visual tokens must not be loss targets')
            answer_tokens=len(processor.tokenizer.encode(r['conversations'][1]['value'],add_special_tokens=False)); seq=len(b['input_ids']); flags=[]
            if r['num_objects']>60: flags.append('objects_gt60')
            if len(r['conversations'][1]['value'])>3500: flags.append('answer_chars_gt3500')
            if not flags: legacy_kept.append(r)
            grid=b['image_grid_thw'][0].tolist(); row={'split':split,'sample_id':r['id'],'num_objects':r['num_objects'],'whole_image': '_whole_' in r['id'],'answer_chars':len(r['conversations'][1]['value']),'answer_tokens':answer_tokens,'supervised_tokens_including_eos':len(target),'masked_prompt_tokens':int((b['labels']==-100).sum()),'prompt_text_tokens':len(processor.tokenizer.encode(r['conversations'][0]['value'].replace('<image>\n',''),add_special_tokens=False)),'visual_tokens':grid[0]*grid[1]*grid[2]//4,'processed_width':grid[2]*14,'processed_height':grid[1]*14,'sequence_tokens':seq,'legacy_filter_reasons':flags,'exceeds_old_config_2048':seq>2048,'fits_sequence_budget':seq<=sequence_budget,'unique_annotation_ids':[o['annotation_id'] for o in r['supervision_objects']],'classes':sorted({o['label'] for o in r['supervision_objects']})}
            rows.append(row)
            if seq<=sequence_budget: kept.append(r)
            if len(saved)<2 and seq<=sequence_budget: saved.append(b)
            if i%100==0: print(split,i,len(records),flush=True)
        batch=collate_fn(saved,pad_token_id=processor.tokenizer.pad_token_id); pad=batch['attention_mask']==0
        if not (batch['labels'][pad]==-100).all(): raise ValueError('Padding contributes to loss')
        summary={'before':coverage(records),'legacy_kept':coverage(legacy_kept),'sequence_budget':sequence_budget,'budget_kept':coverage(kept),'legacy_filter_counts':dict(collections.Counter(flag for row in rows for flag in row['legacy_filter_reasons'])),'legacy_overlap_both':sum(len(row['legacy_filter_reasons'])==2 for row in rows),'legacy_token_filter_actually_present':False,'sequence_min':min(r['sequence_tokens'] for r in rows),'sequence_max':max(r['sequence_tokens'] for r in rows),'full_sequence_over2048':sum(r['exceeds_old_config_2048'] for r in rows),'answer_over1024':sum(r['answer_tokens']>1024 for r in rows),'long_answer_origin':dict(collections.Counter('whole' if r['whole_image'] else 'tile' for r in rows if r['answer_tokens']>1024)),'collated_batch_shape':list(batch['input_ids'].shape),'padding_tokens':int(pad.sum()),'all_masks_and_complete_answers_passed':True}
        all_summary[split]=summary; all_rows.extend(rows)
    (output/'summary.json').write_text(json.dumps(all_summary,indent=2),encoding='utf-8'); (output/'samples.json').write_text(json.dumps(all_rows,indent=2),encoding='utf-8')
    with (output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as f:
        w=csv.DictWriter(f,fieldnames=list(all_rows[0])); w.writeheader(); w.writerows(all_rows)
    return all_summary
if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--sft',required=True); ap.add_argument('--model',required=True); ap.add_argument('--output',required=True); ap.add_argument('--sequence-budget',type=int,default=1536); args=ap.parse_args(); print(json.dumps(audit(args.sft,args.model,args.output,sequence_budget=args.sequence_budget),indent=2))
