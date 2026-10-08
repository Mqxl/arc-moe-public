#!/usr/bin/env python3
"""Create an isolated MTP view with a union token subset or the full draft head.
Existing dense/expert files are linked read-only by convention; the source RT is never modified.
"""
import argparse, struct
from pathlib import Path

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--rt',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--mode',choices=['union','full'],required=True)
    ap.add_argument('--vocab',type=Path,action='append',default=[])
    a=ap.parse_args();src=a.rt.resolve();dst=a.out.resolve()
    if src==dst:ap.error('--out must be a separate directory')
    if a.mode=='union' and not a.vocab:ap.error('union needs at least one --vocab')
    ids=set()
    for path in a.vocab:
        raw=path.read_bytes()
        if not raw or len(raw)%4:ap.error(f'{path}: expected packed int32 token IDs')
        values=[x[0] for x in struct.iter_unpack('<i',raw)]
        if min(values)<0:ap.error(f'{path}: negative token ID')
        ids.update(values)
    dst.mkdir(parents=True,exist_ok=True)
    for name in ['dense.bin','dense.txt','experts.bin']:
        source=src/name;target=dst/name
        if not source.is_file():ap.error(f'missing {source}')
        if target.exists() or target.is_symlink():
            if target.resolve()!=source:ap.error(f'{target} already points elsewhere')
        else:target.symlink_to(source)
    subset=dst/'draft_vocab.bin'
    if a.mode=='full':
        subset.unlink(missing_ok=True)
        print('Full draft head: no subset file; the main model head is reused')
    else:
        temp=subset.with_suffix('.tmp')
        temp.write_bytes(b''.join(struct.pack('<i',i) for i in sorted(ids)))
        temp.replace(subset)
        print(f'Union subset: {len(ids)} token IDs in {subset}')
    return 0
if __name__=='__main__':raise SystemExit(main())
