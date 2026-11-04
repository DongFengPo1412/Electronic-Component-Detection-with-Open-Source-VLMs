"""Read the XML class field independently of reference designators or OCR text."""
import re
import shlex

def match_raw_category(raw_name, pattern_to_id, ignore_patterns):
    try:
        fields = shlex.split(raw_name)
    except ValueError:
        return 0, 'malformed_class'
    if not fields:
        return 0, 'empty_class'
    kind = fields[0].strip().lower()
    # Quoted compound class names stay intact; all later tokens are values, never class evidence.
    for ignore in ignore_patterns:
        if re.search(r'(?<![a-z0-9])'+re.escape(ignore.lower())+r'(?![a-z0-9])',kind):
            return 0, 'ignored_class:' + kind
    for pattern,cid in sorted(pattern_to_id.items(),key=lambda pair:-len(pair[0])):
        if re.search(r'(?<![a-z0-9])'+re.escape(pattern)+r'(?![a-z0-9])',kind):
            return cid,pattern
    return 0,'unmatched_class:' + kind

def audit_annotation_types(coco, manifest, pattern_to_id, ignore_patterns):
    """Trace category semantics back to XML; a valid rectangle is not sufficient evidence."""
    from pathlib import Path
    import xml.etree.ElementTree as ET
    lookup = {}
    missing_xml = []
    for image in manifest:
        path = Path(image['image_path']).with_suffix('.xml')
        if not path.is_file():
            missing_xml.append(str(path)); continue
        for obj in ET.parse(path).getroot().findall('object'):
            box = obj.find('bndbox')
            if box is None: continue
            x,y,X,Y = [float(box.findtext(k)) for k in ('xmin','ymin','xmax','ymax')]
            coords = (max(0,min(image['width']-1,x)), max(0,min(image['height']-1,y)),
                      max(0,min(image['width'],X)), max(0,min(image['height'],Y)))
            name = obj.findtext('name') or ''
            cid,reason = match_raw_category(name,pattern_to_id,ignore_patterns)
            key = (image['image_id'],tuple(round(v,2) for v in coords))
            lookup.setdefault(key,[]).append({'raw_name':name,'expected_category_id':cid,'reason':reason})
    conflicts=[]
    for ann in coco['annotations']:
        box = ann.get('bbox_xyxy')
        if box is None:
            x,y,w,h=ann['bbox']; box=[x,y,x+w,y+h]
        sources = lookup.get((ann['image_id'],tuple(round(v,2) for v in box)),[])
        if not any(source['expected_category_id']==ann['category_id'] for source in sources):
            conflicts.append({'annotation_id':ann['id'],'image_id':ann['image_id'],'assigned_category_id':ann['category_id'],'sources':sources})
    return {'conflicts':conflicts,'missing_xml':missing_xml,'passed':not conflicts and not missing_xml}

def classify_raw_name(raw_name, policy):
    """V3 exact ontology matching. OCR/reference tokens never influence class membership."""
    if not isinstance(raw_name,str) or not raw_name.strip():
        return {'raw_class':'','value':'','decision':'ambiguous','reason':'missing_class'}
    try:
        fields = shlex.split(raw_name)
    except ValueError:
        return {'raw_class':'','value':'','decision':'ambiguous','reason':'malformed_quoting'}
    kind = ' '.join(fields[0].lower().split())
    value = ' '.join(fields[1:])
    if kind in policy['retained_raw_classes']:
        return {'raw_class':kind,'value':value,'decision':'keep','category':policy['retained_raw_classes'][kind],'reason':'explicit_retained_class'}
    if kind in policy['ignored_raw_classes']:
        return {'raw_class':kind,'value':value,'decision':'ignore','reason':policy['ignored_raw_classes'][kind]}
    return {'raw_class':kind,'value':value,'decision':'ambiguous','reason':'unrecognized_class_not_in_ontology'}
