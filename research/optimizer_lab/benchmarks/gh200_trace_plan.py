#!/usr/bin/env python3
"""Measure GPU row hashing/unique/inverse and synthetic gradient aggregation.
Uses genuine FineWeb tokens, but not official BOS packing; not model training.
The merger here uses FP32 accumulation, NOT certified #92 distributed rounding.
It measures the cost of an otherwise missing operation, not numerical parity.
"""
import argparse,hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch,triton
import triton.language as tl
sys.path.insert(0,str(Path(__file__).resolve().parent))
from gh200_table_path import ngram_row_ids,stats,NGRAM_VOCAB_SIZE,NGRAM_DIM

@triton.jit
def hash_rows(TOK,OUT,N:tl.constexpr,PER:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0)*B+tl.arange(0,B);m=i<N;pos=i%PER
    x=tl.load(TOK+i,m,0).to(tl.int32)
    x1=tl.load(TOK+i-1,m&(pos>=1),0).to(tl.int32)
    x2=tl.load(TOK+i-2,m&(pos>=2),0).to(tl.int32)
    mod:tl.constexpr=42301439
    a=(36313*x) ^ (27191*tl.where(pos==1,mod,x1))
    ar=a%mod;ar=tl.where(ar<0,ar+mod,ar)
    a=tl.where(pos==0,mod,ar)
    b=(17351*x) ^ (60961*x1) ^ (45259*x2)
    br=b%mod;br=tl.where(br<0,br+mod,br)
    b=tl.where(pos<2,mod,br)+42301440
    # Match the concatenated per-microbatch CPU function: [bigram, trigram].
    chunk=i//PER;off=i%PER;base=chunk*(2*PER)
    tl.store(OUT+base+off,a,m);tl.store(OUT+base+PER+off,b,m)

@triton.jit
def accumulate_rows(G,INV,ACC,N:tl.constexpr,D:tl.constexpr,B:tl.constexpr):
    i=tl.program_id(0).to(tl.int64)*B+tl.arange(0,B);m=i<N*D
    row=tl.load(INV+i//D,m,0).to(tl.int64)
    g=tl.load(G+i,m,0).to(tl.float32)
    tl.atomic_add(ACC+row*D+i%D,g,m,sem='relaxed')

def plan(tokens_cpu,per):
    x=tokens_cpu.to('cuda',non_blocking=True)
    ids=torch.empty(2*x.numel(),dtype=torch.int32,device='cuda')
    hash_rows[(triton.cdiv(x.numel(),1024),)](x,ids,x.numel(),per,1024)
    rows,inverse=torch.unique(ids,sorted=True,return_inverse=True)
    return rows,inverse,ids

def verify():
    x=torch.randint(0,50304,(8*4096,),dtype=torch.int32,pin_memory=True)
    a,b,c=plan(x,4096)
    expected=torch.cat([ngram_row_ids(z) for z in x.split(4096)])
    torch.testing.assert_close(c.cpu(),expected,rtol=0,atol=0)
    torch.testing.assert_close(a[b],c,rtol=0,atol=0)
    g=torch.randn(61,NGRAM_DIM,device='cuda',dtype=torch.float16)
    inv=torch.randint(0,13,(61,),device='cuda');acc=torch.zeros(13,NGRAM_DIM,device='cuda')
    accumulate_rows[(triton.cdiv(g.numel(),4096),)](g,inv,acc,61,NGRAM_DIM,4096,num_warps=4)
    reference=torch.zeros_like(acc).index_add_(0,inv,g.float())
    torch.testing.assert_close(acc,reference,rtol=1e-5,atol=1e-5)
    return {'hash_ids_bitwise_vs_upstream':True,'inverse_mapping_exact':True,'fp32_merger_vs_torch':True,'flat_indices':'int64 to support more than 2^31 gradient elements'}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8)
    result={'scope':__doc__,'prior_failed_attempt':'FP32 surrogate merge used int32 flattened offsets; late-stage tensor exceeds 2^31 elements. Corrected to int64; failed logs retained.','correctness':verify(),'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'cases':[]}
    data=Path('/home/ubuntu/optimizer-research-cache/fineweb10B/fineweb_train_000001.bin')
    mm=np.memmap(data,mode='r',offset=1024,dtype=np.uint16)
    cases=[('early',131072,2,0),('middle',262144,4,12000000),('late',393216,4,30000000),('taper',327680,4,60000000),('extension',131072,4,75000000)]
    for name,tokens,period,offset in cases:
        inputs=[]
        for j in range(3):
            x=torch.from_numpy(np.array(mm[offset+j*tokens*period:offset+(j+1)*tokens*period],dtype=np.int32)).pin_memory();inputs.append(x)
        times=[];unique=[]
        for i in range(15):
            torch.cuda.synchronize();t=time.perf_counter()
            rows,inverse,ids=plan(inputs[i%3],tokens//8)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
            if i>=3:times.append(elapsed);unique.append(rows.numel())
        # FP32 atomic aggregate (not original distributed rounding), including zero + cast.
        n=rows.numel();grad=torch.randn((inverse.numel(),NGRAM_DIM),device='cuda',dtype=torch.float16)
        acc=torch.empty((n,NGRAM_DIM),device='cuda',dtype=torch.float32)
        output=torch.empty_like(acc,dtype=torch.float16)
        merge=[]
        for i in range(12):
            torch.cuda.synchronize();t=time.perf_counter();acc.zero_()
            accumulate_rows[(triton.cdiv(grad.numel(),4096),)](grad,inverse,acc,inverse.numel(),NGRAM_DIM,4096,num_warps=4)
            output.copy_(acc)
            torch.cuda.synchronize();elapsed=(time.perf_counter()-t)*1000
            if i>=2:merge.append(elapsed)
        case={'name':name,'period':period,'tokens':tokens,'unique_rows':unique[:3],'plan_cpu_to_gpu_hash_unique_inverse':stats(times),'plan_ms_per_step':np.mean(times)/period,'synthetic_fp32_gradient_merge':stats(merge),'merge_ms_per_step':np.mean(merge)/period,'excludes':'disk loading and CPU token staging; gradient generation; Transformer; original distributed rounding'}
        result['cases'].append(case);(a.out/'planner_results.json').write_text(json.dumps(result,indent=2))
        print(name,json.dumps(case),flush=True)
        del grad,acc,output,rows,inverse,ids,inputs;torch.cuda.empty_cache()
    result['completed']=True;(a.out/'planner_results.json').write_text(json.dumps(result,indent=2));print('DONE',flush=True)
if __name__=='__main__':main()
