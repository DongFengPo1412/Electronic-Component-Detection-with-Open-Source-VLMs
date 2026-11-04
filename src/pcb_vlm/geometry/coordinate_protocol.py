"""Explicit project coordinate contracts; syntax acceptance is not model capability evidence."""
import math
import re
from pcb_vlm.config import get_canonical_categories

PROTOCOLS = {f'{order}_{unit}' for order in ('xyxy','yxyx') for unit in ('norm1000','norm1','pixels')}

def to_pixels(coords, width, height, protocol='xyxy_norm1000'):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unsupported coordinate protocol: {protocol}')
    if width <= 0 or height <= 0 or len(coords) != 4 or not all(math.isfinite(v) for v in coords):
        raise ValueError('Nonfinite coordinates or invalid image dimensions')
    order, unit = protocol.split('_')
    a,b,c,d = coords
    x1,y1,x2,y2 = (a,b,c,d) if order == 'xyxy' else (b,a,d,c)
    bound_x,bound_y = (width,height) if unit=='pixels' else ((1000,1000) if unit=='norm1000' else (1,1))
    if not (0 <= x1 < x2 <= bound_x and 0 <= y1 < y2 <= bound_y):
        raise ValueError('Out-of-range, degenerate, or reversed coordinates; no clipping or swapping')
    return [x1/ bound_x*width, y1/bound_y*height, x2/bound_x*width, y2/bound_y*height]

def parse_response(text, width, height, protocol='xyxy_norm1000', allow_qwen_tokens=True):
    if protocol not in PROTOCOLS:
        raise ValueError(f'Unsupported coordinate protocol: {protocol}')
    clean = text.strip()
    for token in ('<|im_end|>', '</s>', '<｜end▁of▁sentence｜>'):
        clean = clean.replace(token, '').strip()
    audit = dict(coordinate_protocol=protocol, raw_candidates=0, valid_geometry=0, valid_categories=0,
                 unknown_categories=0, invalid_geometry=0, duplicate_boxes=0, final_boxes=0)
    if not allow_qwen_tokens and '<|box_start|>' in clean:
        return dict(objects=[],status='parse_failed',detail='Unsupported Qwen tokens for DeepSeek-VL-v1',truncated=False,audit=audit)
    if '<|det|>' in clean or '<|ref|>' in clean:
        return dict(objects=[],status='parse_failed',detail='Unsupported DeepSeek-VL2 syntax for the deployed v1 adapter',truncated=False,audit=audit)
    truncated = any(clean.count(a)>clean.count(b) for a,b in [('[',']'),('(',')'),('<|box_start|>','<|box_end|>')]) or bool(re.search(r'<\|(?:box_start|object_ref_start)[^>]*$',clean))
    num=r'([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)'
    pair=r'\(\s*'+num+r'\s*,\s*'+num+r'\s*\)\s*,?\s*\(\s*'+num+r'\s*,\s*'+num+r'\s*\)'
    quad=r'\[\s*'+num+r'\s*,\s*'+num+r'\s*,\s*'+num+r'\s*,\s*'+num+r'\s*\]'
    label=r'([a-zA-Z_][a-zA-Z_0-9-]*)'
    candidates=[]
    def collect(pattern, label_index=1, coord_start=2):
        for m in re.finditer(pattern,clean):
            candidates.append((m.start(),m.end(),m.group(label_index).lower(),[float(m.group(i)) for i in range(coord_start,coord_start+4)]))
    if allow_qwen_tokens:
        collect(r'<\|object_ref_start\|>\s*'+label+r'\s*<\|object_ref_end\|>\s*<\|box_start\|>\s*'+pair+r'\s*<\|box_end\|>')
        collect(label+r'\s*<\|box_start\|>\s*'+pair+r'\s*<\|box_end\|>')
    collect(r'''\{[^{}]*?["'](?:label|category)["']\s*:\s*["']'''+label+r'''["'][^{}]*?["'](?:bbox|bbox_2d|box)["']\s*:\s*'''+quad+r'[^{}]*?\}')
    for m in re.finditer(r'''\{[^{}]*?["'](?:bbox|bbox_2d|box)["']\s*:\s*'''+quad+r'''[^{}]*?["'](?:label|category)["']\s*:\s*["']'''+label+r'''["'][^{}]*?\}''',clean):
        candidates.append((m.start(),m.end(),m.group(5).lower(),[float(m.group(i)) for i in range(1,5)]))
    collect(r'\b'+label+r'\s*'+quad)
    collect(r'\b'+label+r'\s*'+pair)
    # Bare geometry is recorded as unknown, never assigned a component class.
    for pattern in (pair,quad):
        for m in re.finditer(pattern,clean):
            candidates.append((m.start(),m.end(),'unknown',[float(v) for v in m.groups()]))
    candidates.sort(key=lambda c:(c[0],-c[1]))
    spans=[]; objects=[]; seen=set()
    allowed=set(get_canonical_categories())
    for start,end,label,coords in candidates:
        if any(max(start,a)<min(end,b) for a,b in spans): continue
        spans.append((start,end)); audit['raw_candidates']+=1
        try: box=to_pixels(coords,width,height,protocol)
        except ValueError:
            audit['invalid_geometry']+=1; continue
        audit['valid_geometry']+=1
        if label not in allowed:
            audit['unknown_categories']+=1; continue
        audit['valid_categories']+=1
        key=(label,tuple(box))
        if key in seen: audit['duplicate_boxes']+=1
        seen.add(key)
        objects.append(dict(label=label,bbox_xyxy=box,raw_coordinates=coords,coordinate_protocol=protocol))
    audit['final_boxes']=len(objects)
    no_objects=bool(re.fullmatch(r'(?:none|\[\]|no (?:components?|objects?|targets?)(?: visible| found| present| detected)?)[.]?',clean,re.I))
    rejected=audit['unknown_categories']+audit['invalid_geometry']
    if objects: status='partial_success' if truncated or rejected else 'success'
    elif no_objects and not truncated: status='no_objects_detected'
    elif audit['raw_candidates']: status='parse_filtered'
    else: status='parse_failed'
    return dict(objects=objects,status=status,detail=f'explicit {protocol}; candidates={audit["raw_candidates"]}, accepted={len(objects)}, rejected={rejected}; format mismatch' if status=='parse_failed' else f'explicit {protocol}; {audit}',truncated=truncated,audit=audit)

def generation_finished(ids, eos_ids):
    if isinstance(eos_ids,int): eos_ids=[eos_ids]
    return bool(ids) and ids[-1] in (eos_ids or [])
