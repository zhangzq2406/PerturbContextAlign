#!/usr/bin/env python3
"""Resolve named path tokens in archival JSON configs; no computation is started."""
from __future__ import annotations
import argparse,json,os,re
from pathlib import Path

def expand(value,variables):
    if isinstance(value,dict):return {k:expand(v,variables) for k,v in value.items()}
    if isinstance(value,list):return [expand(v,variables) for v in value]
    if not isinstance(value,str):return value
    def one(m):
        if m[1] not in variables:raise ValueError(f'Missing path variable {m[1]} (use --set {m[1]}=...)')
        return variables[m[1]]
    return re.sub(r'\$\{([A-Z_][A-Z_0-9]*)\}',one,value)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--set',action='append',default=[]);a=p.parse_args()
    variables=dict(os.environ)
    for item in a.set:
        if '=' not in item:p.error('--set must be NAME=value')
        k,v=item.split('=',1);variables[k]=v
    files=sorted(a.input.rglob('*.json')) if a.input.is_dir() else [a.input]
    staged=[]
    for src in files:
        dst=a.output/src.relative_to(a.input) if a.input.is_dir() else a.output
        if dst.exists():raise FileExistsError(f'Refusing to overwrite {dst}')
        staged.append((dst,expand(json.loads(src.read_text()),variables)))
    for dst,obj in staged:dst.parent.mkdir(parents=True,exist_ok=True);dst.write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
    print(f'Resolved {len(staged)} configs; no analysis started. Original frozen hashes are not rewritten.')
if __name__=='__main__':main()
