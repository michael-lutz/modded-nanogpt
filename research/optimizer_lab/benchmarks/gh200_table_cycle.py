#!/usr/bin/env python3
"""Integrated table subsystem benchmark, not a training run.
Full-size host-bound #92 table; actual FineWeb token/hash traces and global stage
batches; GPU planning, FP32 surrogate gradient merge, upstream row Adam, writeback.
No Transformer forward/backward, final loss, or exact eight-rank gradient rounding.
"""
import argparse,gc,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch,triton
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gh200_table_path import host_zeros,copy_rows,copy_state,update,stats,numa_for,correctness,NGRAM_VOCAB_SIZE,NGRAM_DIM
from gh200_trace_plan import plan,verify,accumulate_rows

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--traces',type=int,default=12);ap.add_argument('--profile',action='store_true');a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>200*2**30, 'Need at least 200 GiB available system memory'
    assert 'GH200' in torch.cuda.get_device_name(), 'This benchmark is intended for GH200'
    result={'scope':__doc__,'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'gpu':torch.cuda.get_device_name(),'distinct_traces':a.traces,'trace_offsets':'consecutive windows, wrapping within the 100M-token shard when necessary','correctness':{'table':correctness(),'planner_and_surrogate_merge':verify()},'cases':[],'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}
    t=time.perf_counter();host=host_zeros((NGRAM_VOCAB_SIZE,NGRAM_DIM),torch.bfloat16);hv=host_zeros((NGRAM_VOCAB_SIZE,),torch.float32);hl=host_zeros((NGRAM_VOCAB_SIZE,),torch.int32)
    result['allocation_seconds']=time.perf_counter()-t
    print('ALLOC',result['allocation_seconds'],numa_for(host),flush=True)
    mm=np.memmap('/home/ubuntu/optimizer-research-cache/fineweb10B/fineweb_train_000001.bin',mode='r',offset=1024,dtype=np.uint16)
    h=torch.full((10000,),.95,device='cuda');event=0
    cases=[('early',131072,2,0),('middle',262144,4,12000000),('late',393216,4,30000000),('taper',327680,4,60000000),('extension',131072,4,75000000)]
    for name,tokens,period,offset in cases:
        inputs=[]
        for j in range(a.traces):
            lo=(offset+j*tokens*period) % (len(mm)-tokens*period+1)
            inputs.append(torch.from_numpy(np.array(mm[lo:lo+tokens*period],dtype=np.int32)).pin_memory())
        # Synthetic per-occurrence gradients stand in for real backward output; generation untimed.
        grad=torch.randn(2*tokens*period,NGRAM_DIM,device='cuda',dtype=torch.float16)
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();elapsed=[];phase_samples=[]
        for i in range(36):
            torch.cuda.synchronize();start=time.perf_counter()
            marks=[torch.cuda.Event(enable_timing=True) for _ in range(6)] if a.profile else None
            if marks:marks[0].record()
            rows,inverse,ids=plan(inputs[i%len(inputs)],tokens//8);n=rows.numel()
            if marks:marks[1].record()
            cache=torch.empty(n,NGRAM_DIM,dtype=torch.bfloat16,device='cuda')
            v=torch.empty(n,device='cuda');last=torch.empty(n,dtype=torch.int32,device='cuda')
            local=torch.arange(n,dtype=torch.int32,device='cuda')
            copy_rows(host,cache,rows);copy_state(hv,hl,v,last,rows)
            if marks:marks[2].record()
            acc=torch.zeros(n,NGRAM_DIM,device='cuda')
            accumulate_rows[(triton.cdiv(grad.numel(),4096),)](grad,inverse,acc,inverse.numel(),NGRAM_DIM,4096,num_warps=4)
            merged=acc.to(torch.float16)
            if marks:marks[3].record()
            event+=1;update(cache,v,last,h,merged,local,local,event)
            if marks:marks[4].record()
            copy_rows(cache,host,rows,True);copy_state(v,last,hv,hl,rows,True)
            if marks:marks[5].record()
            torch.cuda.synchronize();ms=(time.perf_counter()-start)*1000
            if i>=6:
                elapsed.append(ms)
                if marks:phase_samples.append([marks[j].elapsed_time(marks[j+1]) for j in range(5)])
        placement=numa_for(host)
        assert placement and all('bind:0' in x and not any(f'N{j}=' in x for j in range(1,9)) for x in placement),placement
        assert bool(torch.isfinite(cache).all()) and bool(torch.isfinite(v).all())
        case={'name':name,'global_tokens_per_step':tokens,'steps_per_table_event':period,'cycle':stats(elapsed),'mean_ms_per_global_step':np.mean(elapsed)/period,'raw_cycle_ms':elapsed,'table_placement':placement,'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated(),'gpu_peak_reserved_bytes':torch.cuda.max_memory_reserved(),'includes':'pinned token upload; GPU hash/unique/inverse; cache allocations; host weights+state fetch; FP32 surrogate gradient aggregation; upstream sparse Adam; host weights+state writeback; final synchronization','excludes':'disk/CPU token staging; actual model forward/backward; official eight-rank rounding; compilation/warmup'}
        if phase_samples:
            case['profile_phase_mean_ms']=dict(zip(['plan','pull_and_alloc','merge','adam','push'],np.mean(phase_samples,axis=0).tolist()))
            print('PROFILE',name,case['profile_phase_mean_ms'],flush=True)
        result['cases'].append(case);(a.out/'integrated_results.json').write_text(json.dumps(result,indent=2))
        print(name,'cycle ms',round(np.mean(elapsed),3),'per global step',round(np.mean(elapsed)/period,3),'GPU peak GiB',round(torch.cuda.max_memory_allocated()/2**30,3),flush=True)
        del inputs,grad,rows,inverse,ids,cache,v,last,local,acc,merged;gc.collect();torch.cuda.empty_cache()
    result['completed']=True;result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(a.out/'integrated_results.json').write_text(json.dumps(result,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
