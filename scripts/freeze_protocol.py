"""Validate and export a frozen protocol without overwriting historical protocols."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from pcb_vlm.provenance import load_protocol

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--protocol', default=None)
    parser.add_argument('--output', required=True, help='New JSON protocol path; existing files are protected')
    args = parser.parse_args()
    protocol = load_protocol(args.protocol)
    with open(args.output, 'x', encoding='utf-8') as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)
    print(protocol['final_protocol_hash'])

if __name__ == '__main__':
    main()
