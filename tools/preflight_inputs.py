#!/usr/bin/env python3
"""Inspect configuration path availability only; never open expression/model arrays."""
from __future__ import annotations
import argparse,json,re
from pathlib import Path

def walk(x,key=''):
    if isinstance(x,dict):
        for k,v in x.items():yield from walk(v,key+'.'+k if key else k)
    elif isinstance(x,list):
        for i,v in enumerate(x):yield from walk(v,f'{key}[{i}]')
    elif isinstance(x,str):yield key,x

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('config',type=Path);p.add_argument('--base',type=Path,default=Path.cwd());a=p.parse_args();unresolved=0;missing=0
    for k,v in walk(json.loads(a.config.read_text())):
        if '${' in v:print('UNRESOLVED',k,v);unresolved+=1;continue
        if not any(w in k.lower() for w in ['path','root','manifest','h5ad','file','cache']):continue
        if ' ' in v or not ('/' in v or v.endswith(('.json','.tsv','.h5ad'))):continue
        q=Path(v).expanduser();q=q if q.is_absolute() else a.base/q
        output=any(w in k.lower() for w in ['output','log_root'])
        state='EXISTS' if q.exists() else ('OUTPUT_NOT_CREATED' if output else 'MISSING')
        print(state,k,str(q));missing+=int(state=='MISSING')
    print(f'unresolved={unresolved}; missing={missing}; numerical arrays opened=0')
    raise SystemExit(1 if unresolved or missing else 0)
if __name__=='__main__':main()
