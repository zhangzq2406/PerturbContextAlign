"""One-command local CPU runner with detached execution and persistent logs."""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import importlib.metadata
import importlib.resources as resources
import json
import os
from pathlib import Path
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from .common import ROOT,OUTBASE,ATOL,pins,sha,write_json,verify_inputs,verify_unchanged,write_table,require


def now():return datetime.now(timezone.utc).isoformat()


def live_status(out, data):
    temp=Path(out)/'status.json.tmp'
    temp.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    temp.replace(Path(out)/'status.json')


def copy_source(out):
    base=resources.files('r3app')
    d=Path(out)/'code/r3app';d.mkdir(parents=True,exist_ok=True)
    for p in base.iterdir():
        if p.is_file() and (p.name.endswith('.py') or p.name=='pins.json'):
            (d/p.name).write_bytes(p.read_bytes())
    evidence=resources.files('r3app').parent.joinpath('evidence')
    if evidence.is_dir():
        dest=Path(out)/'code/evidence';dest.mkdir(exist_ok=True)
        for p in evidence.iterdir():
            if p.is_file():(dest/p.name).write_bytes(p.read_bytes())


def pack(out):
    out=Path(out);dest=out/'R3_review.zip';temporary=out/'R3_review.zip.partial'
    entries=[]
    for p in sorted(out.rglob('*')):
        if not p.is_file() or p in [dest,temporary] or p.name=='output_manifest.tsv':continue
        if p.name.endswith('.tmp'):continue
        entries.append(p)
    # Logs and lifecycle status can change after packaging; label snapshot hashes.
    import pandas as pd
    records=[{'path':str(p.relative_to(out)),'bytes':p.stat().st_size,'sha256':sha(p),
              'scope':'PACKAGING_SNAPSHOT' if (p.name=='status.json' or 'logs' in p.parts) else 'FROZEN_OUTPUT'} for p in entries]
    write_table(out/'output_manifest.tsv',pd.DataFrame(records))
    entries.append(out/'output_manifest.tsv')
    with zipfile.ZipFile(temporary,'x',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in entries:z.write(p,arcname=str(p.relative_to(out)))
    with zipfile.ZipFile(temporary) as z:require(z.testzip() is None,'Review ZIP checksum failure')
    temporary.rename(dest)
    return dest


def worker(root,out,spec=None):
    root=Path(root).resolve();out=Path(out).resolve();spec=pins() if spec is None else spec
    started=time.perf_counter();stage='START';task_context=''
    try:
        require(not out.is_relative_to(root) and not root.is_relative_to(out),'Output overlaps immutable input root')
        for name in ['code','results','geometry','qa','logs']:(out/name).mkdir(parents=True,exist_ok=True)
        copy_source(out)
        contract={
            'version':'R3_OFFLINE_V1_20260925','created_utc':now(),'source_root':str(root),
            'scope':'Six fixed Kaggle same-type seen-drug donor-transfer tasks, 1uM/24h',
            'capture_sha256':spec['capture_sha256'],'method_count_original':22,'selected_direct':6,'selected_baselines':7,
            'tasks':6,'atoms_original':816,'atoms_primary':810,'NK_primary':136,'CD4_primary':134,
            'genes_each_task':3000,'precision':'float32 input -> float64 score;no truth reconstruction',
            'gene_min_n':20,'rank_rule':'average ranks;exact ties;constant correlations NA',
            'mae_rule':'frozen prediction_metrics.condition_metrics:scaled absolute-error mean;then equal queries',
            'geometry_rule':'uncentered cosine;positive vector norms required, as in legacy caller',
            'neighbors':{'k':10,'minimum_group_n':12,'relevance':'linear fractional top10','ties':'full exact block',
                         'random':'truth-specific uniform permutation expectation','self':'excluded'},
            'zero_effect_rule':'stop affected task without cohort substitution, matching actual legacy caller',
            'statistics':'descriptive only;no p-values/CI/bootstrap/pooling as independent observations',
            'paired_gene_difference':'common finite-gene support;retain original marginal means separately',
            'tolerance':{'atol':ATOL,'rtol':0},'new_training':False,'new_encoding':False,'raw_h5ad_read':False,
            'state_predictions_scored':False,'input_numeric_effect_union_read':False,
            'historical_qa':'reused with pinned identity, not claimed as new upstream validation',
            'independent_qa':'fresh frozen inputs;separate numerical code;shared IO declarations only'}
        write_json(out/'execution_contract.json',contract)
        (out/'EXECUTION_CONTRACT.md').write_text('# R3 execution contract\n\n```json\n'+json.dumps(contract,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf-8')
        environment={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
                     'packages':{n:importlib.metadata.version(n) for n in ['numpy','pandas','scipy']},
                     'threads':{k:os.environ.get(k) for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']}}
        write_json(out/'environment.json',environment)
        print('R3_START',now(),flush=True);print('ENVIRONMENT',json.dumps(environment,ensure_ascii=False),flush=True)
        live_status(out,dict(status='RUNNING',stage='INPUT_HASHES',pid=os.getpid(),started_utc=now()))
        stage='INPUT_HASHES'
        manifest=verify_inputs(root,spec);write_table(out/'input_manifest.tsv',manifest)
        print('PINNED_INPUTS_PASS',len(manifest),flush=True)
        # Small old code/config/QA are copied as evidence text only; never executed.
        for record in spec['files']:
            rel=record['path']
            if rel.startswith(('code/','configs/','qa/')):
                dest=out/'code/frozen_source_evidence'/rel;dest.parent.mkdir(parents=True,exist_ok=True)
                shutil.copyfile(root/rel,dest)
        stage='FORMULA_FIXTURES'
        from .fixtures import run as fixtures
        fixture=fixtures(out/'qa/formula_fixtures.json');print('FORMULA_FIXTURES_PASS',fixture['n_checks'],flush=True)
        stage='PRODUCER'
        live_status(out,dict(status='RUNNING',stage=stage,pid=os.getpid()))
        from .producer import run as produce
        produce(root,out,spec)
        stage='INDEPENDENT_QA'
        live_status(out,dict(status='RUNNING',stage=stage,pid=os.getpid()))
        from .independent import run as check
        checked=check(root,out,spec)
        stage='INPUT_UNCHANGED'
        verify_unchanged(manifest)
        write_json(out/'qa/input_unchanged.json',dict(status='PASS',n_files=len(manifest),full_SHA256_before_after=True))
        stage='REPORTS'
        from .report import run as report
        report(out)
        elapsed=time.perf_counter()-started
        scientific=dict(status='PASS',completed_utc=now(),elapsed_seconds=elapsed,
                        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                        independent_qa=checked['status'],inputs_unchanged=True,
                        raw_X_read=False,new_training=False,new_encoding=False,
                        stop_after_R3=True,numeric_qa_coverage=checked['counts'])
        write_json(out/'qa/final_scientific_audit.json',scientific)
        print('R3_SCIENTIFIC_QA_PASS',json.dumps(scientific,ensure_ascii=False),flush=True)
        stage='PACKAGING'
        live_status(out,dict(status='SCIENTIFIC_PASS_PACKAGING',stage=stage,pid=os.getpid()))
        archive=pack(out)
        live_status(out,dict(status='PASS',stage='COMPLETED',completed_utc=now(),pid=os.getpid(),
                             scientific_qa='PASS',result_package=str(archive),package_bytes=archive.stat().st_size,
                             elapsed_seconds=time.perf_counter()-started))
        (out/'exit_code.txt').write_text('0\n')
        print('STATUS=PASS\nRESULT_PACKAGE='+str(archive)+'\nPACKAGE_BYTES='+str(archive.stat().st_size),flush=True)
        return 0
    except BaseException as exc:
        failure=dict(status='FAILED',stage=stage,error=repr(exc),created_utc=now(),
                     elapsed_seconds=time.perf_counter()-started,scientific_results_not_accepted=True)
        traceback.print_exc()
        if not (out/'failure.json').exists():write_json(out/'failure.json',failure)
        (out/'logs/traceback.txt').write_text(traceback.format_exc(),encoding='utf-8')
        live_status(out,failure);(out/'exit_code.txt').write_text('1\n')
        print('STATUS=FAILED\nFAILURE_FILE='+str(out/'failure.json'),flush=True)
        try:
            if not (out/'output_manifest.tsv').exists():
                archive=pack(out);print('DIAGNOSTIC_PACKAGE='+str(archive),flush=True)
        except Exception as packing_exc:print('DIAGNOSTIC_PACK_ERROR',repr(packing_exc),flush=True)
        return 1


def status(base,tail):
    runs=[]
    if Path(base).is_dir():
        for p in Path(base).glob('R3_kaggle_alignment_prediction_*/run_manifest.json'):
            try:
                d=json.loads(p.read_text())
                if d.get('application')=='R3_OFFLINE_RUNNER':runs.append((d['created_utc'],p.parent))
            except (ValueError,OSError,KeyError):pass
    if not runs:print('NO_R3_OFFLINE_RUN_FOUND');return 1
    out=max(runs)[1];print('RUN_DIR='+str(out))
    f=out/'status.json'
    if f.exists():print(f.read_text())
    else:print('STATUS=STARTING')
    log=out/'logs/run.log'
    if log.exists() and tail:
        from collections import deque
        print('===== LOG TAIL =====')
        with log.open(errors='replace') as h:print(''.join(deque(h,maxlen=tail)))
    return 0


def main():
    p=argparse.ArgumentParser(description='R3 cache-only runner. Default: start detached CPU run; --status reads latest run.')
    p.add_argument('--source-root',type=Path,default=Path(ROOT));p.add_argument('--output-base',type=Path,default=Path(OUTBASE))
    p.add_argument('--status',action='store_true');p.add_argument('--tail',type=int,default=40)
    p.add_argument('--self-test',action='store_true',help='Synthetic mathematical fixtures only; no project data accessed')
    p.add_argument('--foreground',action='store_true',help='Run in foreground instead of detached child')
    p.add_argument('--worker-run-dir',type=Path,help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.status:return status(a.output_base,a.tail)
    if a.self_test:
        from .fixtures import run
        print(json.dumps(run(),ensure_ascii=False,indent=2));return 0
    if a.worker_run_dir:
        require((a.worker_run_dir/'run_manifest.json').is_file(),'Worker run manifest missing')
        return worker(a.source_root,a.worker_run_dir)
    for rel in ['metadata/kaggle_atomic_preflight_v1','predictions/kaggle_prediction_v1','representations/kaggle_clean_embeddings_v1/full']:
        require((a.source_root/rel).is_dir(),'Missing original cache directory: '+str(a.source_root/rel))
    ts=datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    out=a.output_base/('R3_kaggle_alignment_prediction_'+ts);out.mkdir(parents=True,exist_ok=False)
    for name in ['code','logs','results','geometry','qa']:(out/name).mkdir()
    write_json(out/'run_manifest.json',dict(application='R3_OFFLINE_RUNNER',version='v1',created_utc=now(),
                                          source_root=str(a.source_root.resolve()),original_launcher=str(Path(sys.argv[0]).resolve())))
    application=Path(sys.argv[0]).resolve();frozen=out/'code/r3_kaggle_offline_v1.pyz'
    shutil.copyfile(application,frozen)
    print('RUN_DIR='+str(out),flush=True)
    if a.foreground:return worker(a.source_root,out)
    logfile=out/'logs/run.log'
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1';env['PYTHONUNBUFFERED']='1'
    for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS','VECLIB_MAXIMUM_THREADS']:
        env[key]='1'
    command=[sys.executable,'-B','-u',str(frozen),'--worker-run-dir',str(out),'--source-root',str(a.source_root)]
    with logfile.open('xb') as h:
        child=subprocess.Popen(command,stdin=subprocess.DEVNULL,stdout=h,stderr=subprocess.STDOUT,
                               start_new_session=True,close_fds=True,env=env,cwd=str(out))
    (out/'run.pid').write_text(str(child.pid)+'\n')
    print('STATUS=STARTED_NOT_COMPLETED\nPID='+str(child.pid)+'\nLOG='+str(logfile)+'\nSTATUS_FILE='+str(out/'status.json')+
          '\nEXPECTED_RESULT_PACKAGE='+str(out/'R3_review.zip'),flush=True)
    print("STATUS_COMMAND="+sys.executable+" -B '"+str(application)+"' --status",flush=True)
    return 0
