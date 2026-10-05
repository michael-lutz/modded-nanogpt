#!/usr/bin/env python3
"""Complete-step GH200 feasibility pilot, NOT a certified #92 reproduction.
Current refactored descendant model; cached CUDA12 FA3 with zero-padded heads;
eight serialized logical ranks, real BOS-packed FineWeb, FP8 training, upstream
ANVIL/Adam and full-size CPU ngram store. HBM control stores only the exact rows
visited by this finite trace, without changing hashes/collisions or dropping rows.
No eight-device reduction-rounding equivalence or convergence claim.
"""
import argparse, dataclasses, hashlib, importlib.util, json, os, sys, time, types
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import triton
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path[:0]=[str(HERE),str(ROOT)]
from gh200_table_path import host_zeros,copy_rows,copy_state,numa_for,NGRAM_VOCAB_SIZE,NGRAM_DIM
from gh200_trace_plan import plan,accumulate_rows


def install_attention_compat():
    root=Path(os.environ.get('FA3_CACHE_ROOT','/home/ubuntu/optimizer-research-cache/track1-record91-hf'))
    p=next(root.rglob('flash_attn_interface.py')).parent
    spec=importlib.util.spec_from_file_location('gh200_cached_fa3',p/'__init__.py',submodule_search_locations=[str(p)])
    fa=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fa;spec.loader.exec_module(fa)
    def padded(q,k,v,**kwargs):
        d=max(q.shape[-1],v.shape[-1]);vd=v.shape[-1]
        q=F.pad(q,(0,d-q.shape[-1]));k=F.pad(k,(0,d-k.shape[-1]));v=F.pad(v,(0,d-v.shape[-1]))
        return fa.flash_attn_varlen_func(q,k,v,**kwargs)[...,:vd]
    name='track_1_short.model.attention'
    path=ROOT/'track_1_short/model/attention.py'
    source=path.read_text();needle='flash_attn_interface = load_flash_attn3()'
    assert source.count(needle)==1
    module=types.ModuleType(name);module.__file__=str(path);module.__package__='track_1_short.model'
    module._gh200_compat=types.SimpleNamespace(flash_attn_varlen_func=padded)
    sys.modules[name]=module
    exec(compile(source.replace(needle,'flash_attn_interface = _gh200_compat'),str(path),'exec'),module.__dict__)
    return {'kernel_revision':p.parents[1].name,'binary_sha256':hashlib.sha256(next(p.glob('*.so')).read_bytes()).hexdigest(),'modification':'zero-pad unequal qk/v heads to same width; preserve explicit softmax scale; slice output to original v width'}


def prepare_batches(a,stage,prefix):
    from track_1_short.data import Shard,cu_seqlens_rows,split_attention_segments
    from track_1_short.sampled_softmax import CandidateBuilder
    data=np.memmap(a.data,mode='r',offset=1024,dtype=np.uint16)
    # Private CPU copy; packing is the upstream Shard algorithm with 8 logical ranks.
    shard=Shard(torch.from_numpy(np.array(data[:30_000_000])),a.ranks)
    builders=[CandidateBuilder(50304,a.candidates,a.tokens) for _ in range(a.ranks)]
    for r,b in enumerate(builders):b.reset(r,a.ranks)
    batches=[]
    for step in range(a.cycles*a.period):
        starts,ends=shard.next_batch(a.tokens,stage.train_max_seq_len)
        perstep=[]
        for r in range(a.ranks):
            buf=torch.cat([shard.tokens[i:j] for i,j in zip(starts[r],ends[r])])
            x=buf[:-1].to(torch.int32);y=buf[1:].to(torch.int64)
            e=torch.tensor(ends[r]);s=torch.tensor(starts[r]);e[-1]-=1
            cum=(e-s).cumsum(0)
            if stage.train_max_seq_len>2560:cum=split_attention_segments(cum,2560)
            cu=torch.full((cu_seqlens_rows(a.tokens),),a.tokens,dtype=torch.int32);cu[0]=0
            assert len(cum)<cu.numel(),len(cum)
            cu[1:len(cum)+1]=cum
            candidate,tp,pp=builders[r].build(a.candidates,y.numpy(),prefix.numpy())
            perstep.append(tuple(t.pin_memory() for t in [x,y,cu,torch.from_numpy(candidate.copy()),torch.from_numpy(tp.copy()),torch.from_numpy(pp.copy())]))
        batches.append(perstep)
    shard._loader_thread.join()
    return batches


class ModelRunner:
    def __init__(self,model,tm,a,batch):
        from track_1_short.sampled_softmax import SampledLoss
        from track_1_short.perf.kernels.transpose import transpose_copy
        self.transpose_copy=transpose_copy;self.model=model;self.tm=tm;self.a=a
        self.x,self.y,self.cu,self.cand,self.tp,self.pp=[t.cuda() for t in batch]
        self.slots=torch.zeros(2*a.tokens,dtype=torch.int32,device='cuda')
        self.sink=torch.zeros(2*a.tokens,NGRAM_DIM,dtype=torch.bfloat16,device='cuda',requires_grad=True)
        self.vgrad=torch.zeros_like(model.value_embeds,dtype=torch.float16)
        self.sample=SampledLoss(torch.empty(a.candidates,768,device='cuda',dtype=torch.float8_e4m3fn),torch.empty(768,a.candidates,device='cuda',dtype=torch.float8_e4m3fn),self.tp,self.pp,torch.empty(50304,dtype=torch.int32,device='cuda'))
        self.arange=torch.arange(a.candidates,dtype=torch.int32,device='cuda')
        self.cfg=tm.get_forward_args(self.sample)
        self.cfg=dataclasses.replace(self.cfg,mtp_weights=self.cfg.mtp_weights.clone(),prefix_weight=self.cfg.prefix_weight.clone())
        self.params=list(model.parameters());self.compiled=torch.compile(model,dynamic=False,fullgraph=True) if a.compile else model
        self.graphs=False
    def stage(self,batch,slots):
        for dest,src in zip([self.x,self.y,self.cu,self.cand,self.tp,self.pp],batch):dest.copy_(src,non_blocking=True)
        self.slots.copy_(slots)
        self.cfg.mtp_weights.copy_(self.tm.mtp_weights);self.cfg.prefix_weight.copy_(self.tm.prefix_weight)
        torch.index_select(self.model.lm_head_f8_col.T.view(torch.uint8),0,self.cand,out=self.sample.rows.view(torch.uint8))
        self.transpose_copy(self.sample.rows.view(torch.uint8),self.sample.rows_t.view(torch.uint8))
        self.sample.vocab_pos.fill_(-1);self.sample.vocab_pos.index_copy_(0,self.cand,self.arange)
    def loss(self):
        return self.compiled(self.x,self.y,self.cu,self.slots,self.cfg,ngram_sink=self.sink,value_embed_grad=self.vgrad).sum()
    def capture(self):
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for j in range(3):
                l=self.loss();torch.autograd.grad(l,self.params+[self.sink],allow_unused=True)
                print('model warmup',j,flush=True)
        torch.cuda.current_stream().wait_stream(stream);torch.cuda.synchronize()
        if self.a.graphs:
            self.fgraph=torch.cuda.CUDAGraph();self.bgraph=torch.cuda.CUDAGraph();pool=torch.cuda.graph_pool_handle()
            with torch.cuda.graph(self.fgraph,pool=pool,stream=stream):self.l=self.loss()
            with torch.cuda.graph(self.bgraph,pool=pool,stream=stream):self.grads=torch.autograd.grad(self.l,self.params+[self.sink],allow_unused=True)
            self.graphs=True
            self.fgraph.replay();torch.cuda.synchronize();captured_loss=self.l.item()
            eager_loss=self.loss().item()
            assert abs(captured_loss-eager_loss)<=max(1e-3,abs(eager_loss)*1e-5),(captured_loss,eager_loss)
            print('GRAPH FORWARD CHECK',captured_loss,eager_loss,flush=True)
        self.vgrad.zero_();torch.cuda.synchronize()
    def forward(self):
        if self.graphs:self.fgraph.replay()
        else:self.l=self.loss()
    def backward(self):
        if self.graphs:self.bgraph.replay()
        else:self.grads=torch.autograd.grad(self.l,self.params+[self.sink],allow_unused=True)
        # Custom CE backward ignores grad_output, so divide the actual gradient,
        # NOT the scalar loss. BF16 summation differs from 8-rank NCCL rounding.
        for p,g in zip(self.params,self.grads):
            if g is None:continue
            if p.grad is None:p.grad=g.clone().mul_(1/self.a.ranks)
            else:p.grad.add_(g,alpha=1/self.a.ranks)
        return self.grads[-1]


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True);ap.add_argument('--mode',choices=['host','hbm'],required=True)
    ap.add_argument('--cycles',type=int,default=8);ap.add_argument('--warmup',type=int,default=3)
    ap.add_argument('--tokens',type=int,default=16384);ap.add_argument('--ranks',type=int,default=8)
    ap.add_argument('--stage',type=int,default=0);ap.add_argument('--period',type=int,default=2);ap.add_argument('--candidates',type=int,default=10240)
    ap.add_argument('--compile',action=argparse.BooleanOptionalAction,default=True);ap.add_argument('--graphs',action=argparse.BooleanOptionalAction,default=True)
    ap.add_argument('--data',type=Path,default=Path('/home/ubuntu/optimizer-research-cache/fineweb10B/fineweb_train_000001.bin'))
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True);started=time.perf_counter()
    torch.set_num_threads(8);torch.manual_seed(0);torch.cuda.set_device(0)
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    if a.mode=='host':assert available>200*2**30
    compat=install_attention_compat()
    from track_1_short.model.gpt import GPT
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from track_1_short.config import TRAINING_STAGES,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT
    from track_1_short.schedule import TrainingSchedule,get_rail_beta
    from track_1_short.training import TrainingManager,value_embed_betas_and_wd_mul
    from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs
    from track_1_short.perf.kernels.mlp import prime_stage_cache
    from track_1_short.ngram_table import NGRAM_LR_MUL,adam_beta2_and_wd_mul
    from track_1_short.perf.kernels.ngram_adam import adam_rows_
    dist.init_process_group('nccl',init_method='file://'+str((a.out/'dist-init').resolve()),rank=0,world_size=1,device_id=torch.device('cuda:0'))
    sched=TrainingSchedule(TRAINING_STAGES,1122,20,torch.device('cuda'),LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
    start_step=sched.boundaries[a.stage][0];stage=TRAINING_STAGES[a.stage]
    model=GPT(50257,11,6,128,768,max(a.tokens,32768),ngram_dim=NGRAM_DIM,world_size=8,device=torch.device('cuda')).cuda()
    model.cast_matrix_weights_bf16();model.train()
    model.ngram_cache=torch.zeros(2*a.tokens*a.ranks*a.period,NGRAM_DIM,dtype=torch.bfloat16,device='cuda')
    prefix=build_prefix_table_bucket(model.vocab_size,0,1);model.prefix_table.copy_(prefix)
    og=AnvilBankGraphs(torch.device('cuda'));tm=TrainingManager(model,sched,bank_update=og.update);tm.advance_schedule(start_step)
    batches=prepare_batches(a,stage,prefix);cycles=[]
    for c in range(a.cycles):
        x=torch.cat([b[0] for step in batches[c*a.period:(c+1)*a.period] for b in step]).pin_memory()
        rows,_,_=plan(x,a.tokens);cycles.append((x,rows))
    universe=torch.unique(torch.cat([rows for _,rows in cycles]))
    if a.mode=='host':
        store=host_zeros((NGRAM_VOCAB_SIZE,NGRAM_DIM),torch.bfloat16);sv=host_zeros((NGRAM_VOCAB_SIZE,),torch.float32);sl=host_zeros((NGRAM_VOCAB_SIZE,),torch.int32)
        def placement():
            lines=numa_for(store);assert lines and all('bind:0' in x and not any(f'N{j}=' in x for j in range(1,9)) for x in lines)
            return True
        placement()
    else:
        store=torch.zeros(universe.numel(),NGRAM_DIM,device='cuda',dtype=torch.bfloat16);sv=torch.zeros(universe.numel(),device='cuda');sl=torch.zeros(universe.numel(),device='cuda',dtype=torch.int32)
        def placement():return None
    h=torch.zeros(10000,device='cuda');event=0
    model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True);prime_stage_cache()
    runner=ModelRunner(model,tm,a,batches[0][0]);runner.stage(batches[0][0],torch.zeros(2*a.tokens,dtype=torch.int32,device='cuda'))
    print('CAPTURE START',flush=True);runner.capture();print('CAPTURE DONE',time.perf_counter()-started,flush=True)
    result={'scope':__doc__,'mode':a.mode,'upstream_commit':'4ea6b937337a4889b8cfe3f38a93d120048d8f71','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'attention':compat,'global_tokens':a.tokens*a.ranks,'logical_ranks':a.ranks,'period':a.period,'stage':a.stage,'cycles':a.cycles,'warmup_cycles':a.warmup,'full_logical_table_bytes':NGRAM_VOCAB_SIZE*NGRAM_DIM*2,'physical_store_bytes':store.numel()*store.element_size(),'trace_union_rows':universe.numel(),'graphs':a.graphs,'compile':a.compile,'startup_through_capture_seconds':time.perf_counter()-started,'measurements':[],'limitations':['refactored descendant, not certified source','padded stock ARM CUDA12 FA3 instead of record mixed-width build','serial gradient accumulation and FP32 table merge, not distributed rounding','prepacked real data and prebuilt sampled candidate IDs outside timer; upload and weight gather inside','short training trajectory, no convergence/validation score','HBM store contains exact finite-trace row union, not a full-capacity HBM allocation']}
    def save(): (a.out/'results.json').write_text(json.dumps(result,indent=2))
    for cycle,(x,expectedrows) in enumerate(cycles):
        torch.cuda.synchronize();beg=time.perf_counter();phase={};marks=[]
        def mark(name):
            e=torch.cuda.Event(enable_timing=True);e.record();marks.append((name,e))
        mark('start');rows,inv,ids=plan(x,a.tokens)
        # HBM maps global row IDs into an exact compact store, not a smaller hash table.
        store_rows=rows if a.mode=='host' else torch.searchsorted(universe,rows).to(torch.int32)
        n=rows.numel();cache=model.ngram_cache[:n];v=torch.empty(n,device='cuda');last=torch.empty(n,device='cuda',dtype=torch.int32);local=torch.arange(n,device='cuda',dtype=torch.int32)
        acc=torch.zeros(n,NGRAM_DIM,device='cuda');copy_rows(store,cache,store_rows);copy_state(sv,sl,v,last,store_rows);mark('plan_pull')
        losses=[];step_times=[]
        for within in range(a.period):
            step=cycle*a.period+within;absolute=start_step+step;tm.advance_schedule(absolute)
            tstep=time.perf_counter()
            model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=(absolute%2==0));mark('refresh')
            for rank in range(a.ranks):
                idx=(within*a.ranks+rank)*2*a.tokens;slots=inv[idx:idx+2*a.tokens].to(torch.int32)
                runner.stage(batches[step][rank],slots);mark('stage')
                runner.forward();mark('forward');g=runner.backward();mark('backward')
                accumulate_rows[(triton.cdiv(g.numel(),4096),)](g,slots,acc,slots.numel(),NGRAM_DIM,4096,num_warps=4);mark('ngram_merge')
                # Keep loss on GPU; synchronize only at end of global step.
                losses.append(runner.l.detach().clone()/a.tokens)
            if within==a.period-1:
                event+=1;beta,wd=adam_beta2_and_wd_mul(absolute);h[event]=beta
                lr=.008*sched.get_lr(absolute)*NGRAM_LR_MUL
                adam_rows_(cache,v,last,h,acc.to(torch.bfloat16),local,local,event,beta2=beta,eps=1e-10,step_size=lr*(1-beta**event)**.5,decay=lr*lr*.005*wd,grad_mul=1/a.ranks,sq_mul=(1-beta)/(a.ranks**2))
                copy_rows(cache,store,store_rows,True);copy_state(v,last,sv,sl,store_rows,True);mark('ngram_update_push')
            opt=tm.optimizer;rail=get_rail_beta(absolute,sched.total_steps);opt.set_rails(fast_beta=rail,fast_weight=1.)
            for p,cfg in opt.param_cfgs.items():
                cfg.lr=cfg.initial_lr*sched.get_lr(absolute)
                if cfg.optim=='anvil':cfg.momentum=rail
                elif cfg.label=='value_embeds':cfg.adam_betas,cfg.wd_mul=value_embed_betas_and_wd_mul(absolute)
            if within==a.period-1:
                opt.adam_update_shard(model.value_embeds,runner.vgrad.mul_(1/a.ranks));runner.vgrad.zero_()
            opt.step(do_adam=absolute%2==1,sparse_update=None,deferred_labels=frozenset());mark('dense_optimizer')
            torch.cuda.synchronize();step_times.append((time.perf_counter()-tstep)*1000)
        wall=(time.perf_counter()-beg)*1000
        for (old,ev0),(name,ev1) in zip(marks,marks[1:]):phase[name]=phase.get(name,0.)+ev0.elapsed_time(ev1)
        lossvals=torch.stack(losses).cpu().tolist();assert np.isfinite(lossvals).all(),lossvals
        finite=bool(torch.isfinite(cache).all()) and bool(torch.isfinite(v).all())
        assert finite
        row={'cycle':cycle,'timed':cycle>=a.warmup,'wall_ms':wall,'ms_per_global_step':wall/a.period,'phase_cuda_ms':phase,'step_ms_without_cycle_pull':step_times,'mean_training_objective':float(np.mean(lossvals)),'microbatch_objectives':lossvals,'table_nonzero_rows':int((cache.abs().sum(1)>0).sum()),'table_rms':float(cache.float().square().mean().sqrt()),'table_gradient_rms':float(acc.square().mean().sqrt()),'finite':finite,'table_cpu_only':placement(),'gpu_peak_allocated_bytes':torch.cuda.max_memory_allocated()}
        result['measurements'].append(row);save();print('CYCLE',json.dumps({k:v for k,v in row.items() if k not in ['microbatch_objectives','phase_cuda_ms']}),flush=True)
        if cycle==0 and a.graphs:
            print('CAPTURE OPTIMIZER',flush=True);og.capture([opt.banks[k] for k in ['qk_bank','vo_bank','mlp_bank']])
    result['completed']=True;result['total_wall_seconds']=time.perf_counter()-started;save();dist.destroy_process_group();print('DONE',flush=True)
if __name__=='__main__':main()
