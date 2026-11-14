"""Fail-closed provenance and protocol checks; no model loading or data mutation."""
import hashlib
import json
from pathlib import Path
import subprocess
import importlib.metadata
import yaml

ROOT = Path(__file__).resolve().parents[2]

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def canonical_map():
    return read_json(ROOT / 'configs/board_canonical_map.json')['mapping']

def physical_id(folder):
    mapping = canonical_map()
    if folder not in mapping:
        raise ValueError(f'Unknown board folder: {folder}')
    return mapping[folder]

def validate_partition(splits, manifest):
    seen_boards, seen_images = set(), set()
    by_id = {x['image_id']: x for x in manifest}
    if len(by_id) != len(manifest):
        raise ValueError('Duplicate image IDs in manifest')
    for name in ('train', 'val', 'test'):
        part = splits['splits'][name]
        ids = part['image_ids']
        if len(ids) != len(set(ids)) or seen_images.intersection(ids):
            raise ValueError(f'Duplicate/leaked image IDs: {name}')
        entries = [by_id[i] for i in ids]
        folders = {e['folder'] for e in entries}
        boards = {physical_id(f) for f in folders}
        if folders != set(part['image_folders']) or boards != set(part['boards']):
            raise ValueError(f'Partition identity mismatch: {name}')
        if seen_boards.intersection(boards):
            raise ValueError(f'Physical board leakage: {name}')
        seen_images.update(ids); seen_boards.update(boards)
    if seen_images != set(by_id):
        raise ValueError('Manifest contains unassigned images')

def stamp_samples(records, name, splits_path, manifest):
    splits = read_json(splits_path)
    validate_partition(splits, manifest)
    by_folder = {x['folder']: x for x in manifest}
    cached_hashes = {}
    def cached_hash(path):
        key = str(path)
        if key not in cached_hashes: cached_hashes[key] = sha256(path)
        return cached_hashes[key]
    coco_path = Path(splits_path).with_name('coco_annotations.json')
    data = read_json(coco_path) if coco_path.is_file() else None
    for r in records:
        entry = by_folder[r['board_folder']]
        if entry['image_id'] not in splits['splits'][name]['image_ids']:
            raise ValueError(f'Sample outside {name}: {r.get("id")}')
        actual_hash = cached_hash(entry['image_path'])
        if actual_hash != entry['sha256']:
            raise ValueError(f'Source image changed: {entry["image_path"]}')
        r.update(image_id=entry['image_id'], physical_board_id=physical_id(entry['folder']),
                 split=name, source_image_sha256=actual_hash,
                 dataset_sha256=cached_hash(splits_path), source_image_path=entry['image_path'],
                 sample_image_sha256=cached_hash(r['image']))
        if data is not None:
            r.update(dataset_version=data.get('info', {}).get('version', 'unknown'),
                     annotations_sha256=cached_hash(coco_path),
                     classes_sha256=hashlib.sha256(json.dumps(data['categories'], sort_keys=True).encode()).hexdigest())

def validate_samples(records, name, splits_path):
    if not records:
        raise ValueError(f'Empty {name} samples')
    splits = read_json(splits_path)
    manifest = read_json(Path(splits_path).with_name('images_manifest.json'))
    validate_partition(splits, manifest)
    by_id = {x['image_id']: x for x in manifest}
    allowed = splits['splits'][name]
    seen = set()
    for r in records:
        required = ('image_id','board_folder','physical_board_id','split','source_image_sha256',
                    'dataset_sha256','source_image_path','sample_image_sha256')
        if any(k not in r for k in required):
            raise ValueError(f'Incomplete sample provenance: {r.get("id")}')
        entry = by_id.get(r['image_id'])
        if not entry or r['image_id'] not in allowed['image_ids'] or r['board_folder'] != entry['folder']:
            raise ValueError(f'Invalid image membership: {r.get("id")}')
        board = physical_id(r['board_folder'])
        if r['physical_board_id'] != board or board not in allowed['boards'] or r['split'] != name:
            raise ValueError(f'Physical split conflict: {r.get("id")}')
        if r['dataset_sha256'] != sha256(splits_path):
            raise ValueError('SFT split hash mismatch')
        if Path(r['source_image_path']).resolve() != Path(entry['image_path']).resolve():
            raise ValueError('Source image path mismatch')
        if r['source_image_sha256'] != entry['sha256'] or sha256(entry['image_path']) != entry['sha256']:
            raise ValueError('Source image content mismatch')
        if sha256(r['image']) != r['sample_image_sha256']:
            raise ValueError('SFT crop content mismatch')
        if r.get('id') in seen:
            raise ValueError('Duplicate SFT sample ID')
        seen.add(r.get('id'))
    coco_path = Path(splits_path).with_name('coco_annotations.json')
    if coco_path.is_file() and read_json(coco_path).get('info', {}).get('version') == '3.0-semantic':
        data = read_json(coco_path)
        class_hash = hashlib.sha256(json.dumps(data['categories'], sort_keys=True).encode()).hexdigest()
        for r in records:
            if r.get('dataset_version') != '3.0-semantic' or r.get('annotations_sha256') != sha256(coco_path) or r.get('classes_sha256') != class_hash:
                raise ValueError('V3 sample annotation/category identity mismatch')
    return sorted({r['physical_board_id'] for r in records})

def model_identity(path):
    root = Path(path)
    if not root.is_dir():
        raise FileNotFoundError(f'Formal runs require a local pinned model directory: {path}')
    files = sorted(f for f in root.rglob('*') if f.is_file() and f.suffix in ('.json','.safetensors','.bin'))
    if not any(f.suffix in ('.safetensors','.bin') for f in files):
        raise ValueError(f'No model weights: {path}')
    return hashlib.sha256(json.dumps([(str(f.relative_to(root)), sha256(f)) for f in files]).encode()).hexdigest()

def verify_adapter(path, splits_path, base_model=None):
    root = Path(path)
    if not root.is_dir() or not any(root.glob('adapter_model.*')):
        raise FileNotFoundError(f'Missing LoRA weights: {path}')
    meta = read_json(root / 'adapter_metadata.json')
    required = ('split_sha256','train_sft_sha256','train_boards','base_model_sha256',
                'training_step','training_config','coordinate_protocol','classes','git_commit','dependencies','train_sft_path')
    if any(not meta.get(k) for k in required):
        raise ValueError('Incomplete LoRA provenance; historical use only')
    if meta['split_sha256'] != sha256(splits_path):
        raise ValueError('LoRA split hash mismatch')
    from pcb_vlm.config import validate_category_consistency
    validate_category_consistency(meta['classes'])
    samples = read_json(meta['train_sft_path'])
    if sha256(meta['train_sft_path']) != meta['train_sft_sha256']:
        raise ValueError('LoRA SFT content mismatch')
    boards = validate_samples(samples, 'train', splits_path)
    if set(boards) != set(meta['train_boards']):
        raise ValueError('LoRA training board metadata mismatch')
    if base_model and model_identity(base_model) != meta['base_model_sha256']:
        raise ValueError('LoRA base model mismatch')
    return meta

def load_protocol(path=None):
    path = Path(path or ROOT / 'configs/frozen_protocol.yaml')
    data = yaml.safe_load(path.read_text(encoding='utf-8'))
    split_path = Path(data['dataset_canonical_dir']) / 'splits.json'
    splits = read_json(split_path)
    validate_partition(splits, read_json(split_path.with_name('images_manifest.json')))
    if data['test_split']['image_ids'] != splits['splits']['test']['image_ids'] or set(data['test_split']['boards']) != set(splits['splits']['test']['boards']):
        raise ValueError('Frozen test split differs from canonical split')
    from pcb_vlm.config import validate_category_consistency
    validate_category_consistency(data['categories'])
    from pcb_vlm.config import get_category_definitions
    coco_path = split_path.with_name('coco_annotations.json')
    coco = read_json(coco_path)
    definitions = get_category_definitions()
    if {(c['id'], c['name']) for c in coco['categories']} != {(c['id'], c['name']) for c in definitions}:
        raise ValueError('Canonical annotation category IDs differ from classes.yaml')
    data['category_definitions'] = [{k:c[k] for k in ('id','name','chinese_name')} for c in definitions]
    actual_ann_sha = sha256(coco_path)
    actual_split_sha = sha256(split_path)
    if 'annotations_sha256' in data and data['annotations_sha256'] != actual_ann_sha:
        raise ValueError('Frozen protocol annotations_sha256 mismatch')
    if 'split_sha256' in data and data['split_sha256'] != actual_split_sha:
        raise ValueError('Frozen protocol split_sha256 mismatch')
    already_enriched = 'annotations_sha256' in data and 'split_sha256' in data and 'final_protocol_hash' in data
    recorded_hash = data.get('final_protocol_hash')
    data['annotations_sha256'] = actual_ann_sha
    data['split_sha256'] = actual_split_sha
    data['generation'].setdefault('max_pixels', 602112)
    data['generation'].setdefault('repetition_penalty', 1.25)
    data.setdefault('coordinate_protocols', {'qwen': 'xyxy_norm1000', 'deepseek': 'yxyx_norm1000'})
    from pcb_vlm.geometry.coordinate_protocol import PROTOCOLS
    if any(value not in PROTOCOLS for value in data['coordinate_protocols'].values()):
        raise ValueError('Unsupported explicit coordinate protocol')
    data.pop('final_protocol_hash', None)
    computed_hash = hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()
    if already_enriched and recorded_hash != computed_hash:
        raise ValueError('Frozen protocol final_protocol_hash mismatch; protocol file may be modified')
    data['final_protocol_hash'] = computed_hash
    return data

def code_identity():
    return subprocess.check_output(['git','rev-parse','HEAD'], cwd=ROOT, text=True).strip()

def source_identity():
    paths = sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "src/pcb_vlm").rglob("*.py"))
    return hashlib.sha256(json.dumps([(str(f.relative_to(ROOT)), sha256(f)) for f in paths]).encode()).hexdigest()

def dependencies():
    return {name: importlib.metadata.version(name) for name in ('torch','transformers','peft','numpy','pycocotools')}
