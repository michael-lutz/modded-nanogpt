#!/usr/bin/env python3
"""Full-capacity Grace-host table service benchmark; NOT full #92 training.

Actual FineWeb contiguous token traces and upstream hashes, 8 logical ranks, stage
batch sizes and sparse-update cadence. Not the exact BOS-aligned official loader.
Gradients are synthetic already-aggregated rows; forward/backward, gradient merge,
sampled softmax, communication and convergence are not measured. Includes a checked
GPU cache pull -> unmodified upstream sparse Adam -> host writeback implementation.
Pinned CPU storage is accessed directly by Triton (NVLink-C2C on GH200).
"""
import argparse, ctypes, gc, hashlib, json, mmap, os, platform, statistics, subprocess, sys, time
from pathlib import Path
import numpy as np
import torch
import triton
import triton.language as tl
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from track_1_short.ngram_table import ngram_row_ids, NGRAM_VOCAB_SIZE, NGRAM_DIM
from track_1_short.perf.kernels.ngram_adam import adam_rows_

@triton.jit(do_not_specialize=["N"])
def row_copy(SRC,DST,ROWS,N,D:tl.constexpr,SCATTER:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B)
    valid=x<N*D
    slot=x//D; col=x%D
    row=tl.load(ROWS+slot,valid,0).to(tl.int64)
    indexed=row*D+col
    if SCATTER:
        v=tl.load(SRC+x,valid,0);tl.store(DST+indexed,v,valid)
    else:
        v=tl.load(SRC+indexed,valid,0);tl.store(DST+x,v,valid)

@triton.jit(do_not_specialize=["N"])
def state_copy(SV,SL,DV,DL,ROWS,N,SCATTER:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);valid=i<N
    r=tl.load(ROWS+i,valid,0).to(tl.int64)
    if SCATTER:
        v=tl.load(SV+i,valid,0);l=tl.load(SL+i,valid,0)
        tl.store(DV+r,v,valid);tl.store(DL+r,l,valid)
    else:
        v=tl.load(SV+r,valid,0);l=tl.load(SL+r,valid,0)
        tl.store(DV+i,v,valid);tl.store(DL+i,l,valid)

def copy_rows(src,dst,rows,scatter=False):
    n=rows.numel();row_copy[(triton.cdiv(n*NGRAM_DIM,4096),)](src,dst,rows,n,NGRAM_DIM,scatter,4096,num_warps=4)

def copy_state(sv,sl,dv,dl,rows,scatter=False):
    n=rows.numel();state_copy[(triton.cdiv(n,256),)](sv,sl,dv,dl,rows,n,scatter,256)

def update(p,v,l,h,g,rows,entry,event,beta=.95):
    adam_rows_(p,v,l,h,g,rows,entry,event,beta2=beta,eps=1e-8,step_size=.002,decay=.0001,grad_mul=1.,sq_mul=1.-beta)

_HOST_BUFFERS=[]
def host_zeros(shape,dtype):
    """Explicit node-0 CPU DRAM, including after CUDA registration changes policy."""
    count=int(np.prod(shape));nbytes=count*torch.empty((),dtype=dtype).element_size()
    page=os.sysconf('SC_PAGE_SIZE');mapped=(nbytes+page-1)//page*page
    buf=mmap.mmap(-1,mapped,flags=mmap.MAP_PRIVATE|mmap.MAP_ANONYMOUS)
    x=torch.frombuffer(buf,dtype=dtype,count=count).view(shape)
    lib=ctypes.CDLL('libnuma.so.1',use_errno=True)
    lib.mbind.argtypes=[ctypes.c_void_p,ctypes.c_ulong,ctypes.c_int,ctypes.POINTER(ctypes.c_ulong),ctypes.c_ulong,ctypes.c_uint]
    mask=ctypes.c_ulong(1)
    def bind(flags):
        rc=lib.mbind(x.data_ptr(),mapped,2,ctypes.byref(mask),64,flags)
        if rc: raise OSError(ctypes.get_errno(),'mbind CPU node 0 failed')
    bind(0);x.zero_()
    err=torch.cuda.cudart().cudaHostRegister(x.data_ptr(),mapped,3)
    if int(err)!=0: raise RuntimeError(f'cudaHostRegister {err}')
    # CUDA's registration can replace the policy with preferred; bind it again.
    bind(2)
    _HOST_BUFFERS.append((x,buf,mapped))
    return x

def correctness():
    torch.manual_seed(31)
    n=29;d=NGRAM_DIM
    host=torch.randn(n,d,dtype=torch.bfloat16,pin_memory=True)
    hv=torch.rand(n,dtype=torch.float32,pin_memory=True);hl=torch.zeros(n,dtype=torch.int32,pin_memory=True)
    ref=host.cuda();rv=hv.cuda();rl=hl.cuda()
    h=torch.full((120,),.95,device='cuda');h[7:]=.9025
    untouched=host[28].clone()
    for event,ids in [(1,[0,16,4,7]),(2,[16,1,7,8]),(7,[0,1,3,4]),(8,[0,3,7,16]),(100,[0,1,16,4])]:
        idx=torch.tensor(ids,dtype=torch.int32,device='cuda');m=len(ids)
        p=torch.empty(m,d,dtype=torch.bfloat16,device='cuda');v=torch.empty(m,device='cuda');l=torch.empty(m,dtype=torch.int32,device='cuda')
        loc=torch.arange(m,dtype=torch.int32,device='cuda');entry=torch.zeros(n,dtype=torch.int32,device='cuda');entry[idx.long()]=loc
        g=torch.randn_like(p);g[0].zero_()
        copy_rows(host,p,idx);copy_state(hv,hl,v,l,idx)
        beta=float(h[event])
        update(p,v,l,h,g,loc,loc,event,beta)
        copy_rows(p,host,idx,True);copy_state(v,l,hv,hl,idx,True)
        update(ref,rv,rl,h,g,idx,entry,event,beta)
        torch.cuda.synchronize()
        torch.testing.assert_close(host,ref.cpu(),rtol=0,atol=0)
        torch.testing.assert_close(hv,rv.cpu(),rtol=0,atol=0)
        torch.testing.assert_close(hl,rl.cpu(),rtol=0,atol=0)
    torch.testing.assert_close(host[28],untouched,rtol=0,atol=0)
    return {'bitwise_vs_upstream_gpu':True,'events':[1,2,7,8,100],'tested':['changed row subsets','missed-decay replay','beta transition','zero gradient','untouched row','weights and all moment/timestamp state'],'not_tested':['full eight-rank numerical equivalence','real gradient reduction','training convergence']}

def stats(x):
    return {'mean_ms':float(np.mean(x)),'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'min_ms':float(np.min(x)),'max_ms':float(np.max(x))}

def numa_for(t):
    address=t.data_ptr()
    for line in Path('/proc/self/maps').read_text().splitlines():
        a,b=[int(x,16) for x in line.split()[0].split('-')]
        if a<=address<b:
            key=f'{a:x}'
            return [z for z in Path('/proc/self/numa_maps').read_text().splitlines() if z.split()[0]==key]
    return []

def prepare_trace(mm,offset,tokens,period):
    start=time.perf_counter(); chunks=[]; per=tokens//8
    for step in range(period):
        for rank in range(8):
            lo=offset+step*tokens+rank*per
            x=torch.from_numpy(np.array(mm[lo:lo+per],dtype=np.int32))
            chunks.append(ngram_row_ids(x).numpy())
    flat=np.concatenate(chunks)
    ids=np.unique(flat)
    elapsed=(time.perf_counter()-start)*1000
    return ids,elapsed,int(flat.size)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--cycles',type=int,default=30);ap.add_argument('--traces',type=int,default=3);ap.add_argument('--threads',type=int,default=8)
    ap.add_argument('--data',type=Path,default=Path('/home/ubuntu/optimizer-research-cache/fineweb10B/fineweb_train_000001.bin'))
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(args.threads)
    result={'scope':__doc__,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'torch':torch.__version__,'triton':triton.__version__,'python':platform.python_version(),'cpu_threads':args.threads,'gpu':torch.cuda.get_device_name(),'source_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'correctness':correctness(),'cases':[]}
    def save(): (args.out/'results.json').write_text(json.dumps(result,indent=2))
    print('CORRECTNESS',json.dumps(result['correctness']),flush=True);save()
    meminfo=Path('/proc/meminfo').read_text();result['memory_before']=meminfo
    # Refuse if available system memory is insufficient, allowing a wide headroom.
    available=int(next(x.split()[1] for x in meminfo.splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>200*2**30,f'Not enough headroom: {available}'
    print('Allocating and zero-touching full 129.95GB pinned host table',flush=True)
    t=time.perf_counter()
    host=host_zeros((NGRAM_VOCAB_SIZE,NGRAM_DIM),torch.bfloat16)
    hv=host_zeros((NGRAM_VOCAB_SIZE,),torch.float32)
    hl=host_zeros((NGRAM_VOCAB_SIZE,),torch.int32)
    result['host_allocation_seconds']=time.perf_counter()-t
    result['host_table_bytes']=host.numel()*host.element_size()
    result['host_row_state_bytes']=hv.numel()*8
    result['host_table_numa']=numa_for(host)
    result['memory_after_allocation']=Path('/proc/meminfo').read_text()
    print('ALLOCATED',result['host_allocation_seconds'],'seconds','NUMA',result['host_table_numa'],flush=True);save()
    header=np.fromfile(args.data,dtype=np.int32,count=256);assert header[0]==20240520
    mm=np.memmap(args.data,mode='r',offset=1024,dtype=np.uint16,shape=(int(header[2]),))
    result['data']={'file':args.data.name,'header_tokens':int(header[2]),'header_sha256':hashlib.sha256(header.tobytes()).hexdigest(),'trace_note':'consecutive real FineWeb tokens; hash each logical-rank microbatch separately; no exact BOS alignment/packing'}
    h=torch.full((10000,),.95,device='cuda');event=0
    cases=[('early',131072,2,0),('middle',262144,4,12000000),('late',393216,4,30000000),('taper',327680,4,60000000),('extension',131072,4,75000000)]
    for name,tokens,period,offset in cases:
        torch.cuda.reset_peak_memory_stats()
        traces=[];plans=[]
        for j in range(args.traces):
            ids,plan,occurrences=prepare_trace(mm,offset+j*tokens*period,tokens,period)
            plans.append(plan);rows=torch.from_numpy(ids.copy()).cuda();n=rows.numel()
            cache=torch.empty((n,NGRAM_DIM),device='cuda',dtype=torch.bfloat16)
            v=torch.empty(n,device='cuda');last=torch.empty(n,device='cuda',dtype=torch.int32)
            local=torch.arange(n,device='cuda',dtype=torch.int32)
            g=torch.randn((n,NGRAM_DIM),device='cuda',dtype=torch.float16)
            traces.append((rows,cache,v,last,local,g))
        torch.cuda.synchronize()
        case={'name':name,'tokens_per_global_step':tokens,'steps_per_table_event':period,'logical_ranks':8,'traces':args.traces,'unique_rows':[t[0].numel() for t in traces],'occurrences_per_cycle':occurrences,'planning_serial_cpu':stats(plans),'planning_ms_per_step':float(np.mean(plans))/period,'methods':{}}
        print('CASE',name,'unique rows',case['unique_rows'],'planning ms/cycle',plans,flush=True)
        # Compile and warm all transfer paths. All rows are initialized; no demand-zero shortcut.
        for rows,cache,v,last,local,g in traces:
            copy_rows(host,cache,rows);copy_state(hv,hl,v,last,rows)
            event+=1;update(cache,v,last,h,g,local,local,event)
            copy_rows(cache,host,rows,True);copy_state(v,last,hv,hl,rows,True)
        torch.cuda.synchronize()
        # Each method's first 6 iterations are excluded as warmup. No prefetch overlap is assumed.
        for method in ['host_pull_only','hbm_cache_update_only','host_pull_update_push']:
            measurements=[];device=[]
            for i in range(args.cycles+6):
                rows,cache,v,last,local,g=traces[i%len(traces)]
                begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize();t=time.perf_counter();begin.record()
                if method!='hbm_cache_update_only':
                    copy_rows(host,cache,rows);copy_state(hv,hl,v,last,rows)
                if method!='host_pull_only':
                    event+=1;update(cache,v,last,h,g,local,local,event)
                if method=='host_pull_update_push':
                    copy_rows(cache,host,rows,True);copy_state(v,last,hv,hl,rows,True)
                end.record();end.synchronize();elapsed=(time.perf_counter()-t)*1000
                if i>=6: measurements.append(elapsed);device.append(begin.elapsed_time(end))
            case['methods'][method]={'synced_wall':stats(measurements),'cuda_event':stats(device),'raw_wall_ms':measurements,'raw_device_ms':device,'mean_wall_ms_per_global_step':statistics.mean(measurements)/period}
            print(name,method,'wall ms/cycle',round(statistics.mean(measurements),3),'per step',round(statistics.mean(measurements)/period,3),flush=True)
        case['host_table_numa_after']=numa_for(host)
        print('NUMA AFTER',name,case['host_table_numa_after'],flush=True)
        assert all('bind:0' in x and not any(f'N{i}=' in x for i in range(1,9)) for x in case['host_table_numa_after'])
        case['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();case['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
        case['roundtrip_effective_GBps']=np.mean(case['unique_rows'])*NGRAM_DIM*2*2/(case['methods']['host_pull_update_push']['cuda_event']['mean_ms']*1e6)
        case['serial_table_plus_planning_ms_per_step']=case['planning_ms_per_step']+case['methods']['host_pull_update_push']['mean_wall_ms_per_global_step']
        result['cases'].append(case);save()
        del traces,rows,cache,v,last,local,g;gc.collect();torch.cuda.empty_cache()
    result['finished_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());result['finite_sample']=bool(torch.isfinite(host[0:10]).all());save()
    print('DONE',args.out/'results.json',flush=True)
if __name__=='__main__':main()
