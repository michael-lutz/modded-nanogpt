#!/usr/bin/env python3
"""Full 1194-step single-GH200 #92-descendant adaptation; not certified #360.
Full CPU-resident 129.95GB ngram table, 8 serial logical ranks, full stage schedule,
upstream ANVIL/Adam/tail averaging, BOS training packing, 10,485,760-token validation.
Differences: padded ARM CUDA12 FA3; serial dense-gradient averaging; FP32 ngram
merge; FP8 delayed amax shared across microbatches; corrected descendant canon mask.
"""
import time
PROCESS_START = time.perf_counter()
import argparse, dataclasses, gc, hashlib, json, os, sys, traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import triton
ROOT = Path(__file__).resolve().parents[3]
BENCH = ROOT / 'research/optimizer_lab/benchmarks'
sys.path[:0] = [str(ROOT), str(BENCH)]
from gh200_training_smoke import install_attention_compat
from gh200_table_path import host_zeros, copy_rows, copy_state, numa_for, NGRAM_VOCAB_SIZE, NGRAM_DIM
from gh200_trace_plan import hash_rows, accumulate_rows


def emit(event, **fields):
    print(json.dumps(dict(event=event, utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **fields)), flush=True)


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


class DataStream:
    """Upstream packing, serial logical-rank candidate streams, bounded lookahead."""
    def __init__(self, directory, schedule, prefix):
        from track_1_short.data import Shard, _load_data_shard
        from track_1_short.sampled_softmax import CandidateBuilder
        self.Shard, self.load = Shard, _load_data_shard
        self.paths = iter(sorted(directory.glob('fineweb_train_*.bin')))
        self.shard = None
        self.schedule, self.prefix = schedule, prefix.numpy()
        self.builders = [CandidateBuilder(50304, 24576, 49152) for _ in range(8)]
        for rank, b in enumerate(self.builders):
            b.reset(rank, 8)
        self.loaded = []

    def next(self, step):
        from track_1_short.data import cu_seqlens_rows, split_attention_segments
        from track_1_short.sampled_softmax import candidate_count_at
        stage, _ = self.schedule.lookup(step)
        T, P = stage.batch_size // 8, candidate_count_at(self.schedule, step)
        while True:
            if self.shard is None:
                path = next(self.paths)
                self.shard = self.Shard(self.load(path), 8)
                self.loaded.append(path.name)
                emit('data_shard', file=path.name)
            try:
                starts, ends = self.shard.next_batch(T, stage.train_max_seq_len)
                break
            except StopIteration:
                self.shard._loader_thread.join()
                self.shard = None
        micros = []
        for rank in range(8):
            buf = torch.cat([self.shard.tokens[i:j] for i, j in zip(starts[rank], ends[rank])])
            x, y = buf[:-1].to(torch.int32), buf[1:].to(torch.int64)
            ee, ss = torch.tensor(ends[rank]), torch.tensor(starts[rank])
            ee[-1] -= 1
            cumulative = (ee - ss).cumsum(0)
            if stage.train_max_seq_len > 2560:
                cumulative = split_attention_segments(cumulative, 2560)
            cu = torch.full((cu_seqlens_rows(T),), T, dtype=torch.int32)
            cu[0] = 0
            assert len(cumulative) < cu.numel()
            cu[1:len(cumulative)+1] = cumulative
            if P:
                cand, tp, pp = self.builders[rank].build(P, y.numpy(), self.prefix)
                extra = [torch.from_numpy(z.copy()) for z in (cand, tp, pp)]
            else:
                extra = [torch.empty(0, dtype=torch.int64)] * 3
            micros.append(tuple(z.pin_memory() for z in [x, y, cu, *extra]))
        packed = torch.cat([m[0] for m in micros]).pin_memory()
        return dict(step=step, tokens=T, candidates=P, micros=micros, packed=packed)

    def cycle(self, steps):
        return [self.next(s) for s in steps]

    def close(self):
        if self.shard is not None:
            self.shard._loader_thread.join()


class Runner:
    def __init__(self, model, tm, batch, vgrad, pool):
        from track_1_short.sampled_softmax import SampledLoss
        from track_1_short.perf.kernels.transpose import transpose_copy
        self.model, self.tm, self.vgrad, self.pool = model, tm, vgrad, pool
        self.transpose_copy = transpose_copy
        T, P = batch['tokens'], batch['candidates']
        self.T, self.P = T, P
        self.x, self.y, self.cu, self.cand, self.tp, self.pp = [z.cuda() for z in batch['micros'][0]]
        self.slots = torch.zeros(2*T, dtype=torch.int32, device='cuda')
        self.sink = torch.zeros(2*T, NGRAM_DIM, dtype=torch.bfloat16, device='cuda', requires_grad=True)
        self.sample = None
        if P:
            self.sample = SampledLoss(torch.empty(P,768,device='cuda',dtype=torch.float8_e4m3fn),
                torch.empty(768,P,device='cuda',dtype=torch.float8_e4m3fn), self.tp, self.pp,
                torch.empty(50304,dtype=torch.int32,device='cuda'))
            self.arange = torch.arange(P,dtype=torch.int32,device='cuda')
        cfg = tm.get_forward_args(self.sample)
        self.cfg = dataclasses.replace(cfg, mtp_weights=cfg.mtp_weights.clone(), prefix_weight=cfg.prefix_weight.clone())
        self.params = list(model.parameters())
        self.compiled = torch.compile(model, dynamic=False, fullgraph=True)
        self.stage(batch['micros'][0], self.slots)

    def stage(self, micro, slots):
        for dest, src in zip([self.x,self.y,self.cu,self.cand,self.tp,self.pp], micro):
            dest.copy_(src, non_blocking=True)
        self.slots.copy_(slots)
        self.cfg.mtp_weights.copy_(self.tm.mtp_weights)
        self.cfg.prefix_weight.copy_(self.tm.prefix_weight)
        if self.P:
            torch.index_select(self.model.lm_head_f8_col.T.view(torch.uint8),0,self.cand,out=self.sample.rows.view(torch.uint8))
            self.transpose_copy(self.sample.rows.view(torch.uint8),self.sample.rows_t.view(torch.uint8))
            self.sample.vocab_pos.fill_(-1)
            self.sample.vocab_pos.index_copy_(0,self.cand,self.arange)

    def loss(self):
        return self.compiled(self.x,self.y,self.cu,self.slots,self.cfg,
                            ngram_sink=self.sink,value_embed_grad=self.vgrad).sum()

    def capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                torch.autograd.grad(self.loss(), self.params+[self.sink], allow_unused=True)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.fgraph, self.bgraph = torch.cuda.CUDAGraph(), torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.fgraph,pool=self.pool,stream=stream):
            self.l = self.loss()
        with torch.cuda.graph(self.bgraph,pool=self.pool,stream=stream):
            self.grads = torch.autograd.grad(self.l,self.params+[self.sink],allow_unused=True)
        self.fgraph.replay()
        captured = self.l.item()
        eager = self.loss().item()
        assert abs(captured-eager) <= max(1e-3,abs(eager)*1e-5), (captured,eager)
        self.vgrad.zero_()
        emit('graph_check', tokens=self.T, candidates=self.P, captured=captured, eager=eager)

    def forward_backward(self):
        self.fgraph.replay()
        self.bgraph.replay()
        for param, grad in zip(self.params,self.grads):
            if grad is None:
                continue
            if param.grad is None:
                param.grad = grad.clone().mul_(1/8)
            else:
                param.grad.add_(grad,alpha=1/8)
        return self.grads[-1]


def configure_optimizer(tm, step):
    from track_1_short.schedule import get_rail_beta
    from track_1_short.training import value_embed_betas_and_wd_mul
    from track_1_short.optim.anvil import RAIL_ENGAGE_STEP,RAIL_FAST_BETA,RAIL_FAST_WEIGHT
    opt = tm.optimizer
    rail = get_rail_beta(step,tm.schedule.total_steps)
    opt.set_rails(fast_beta=RAIL_FAST_BETA if step>=RAIL_ENGAGE_STEP else rail,
                  fast_weight=RAIL_FAST_WEIGHT if step>=RAIL_ENGAGE_STEP else 1.)
    for p,cfg in opt.param_cfgs.items():
        cfg.lr = cfg.initial_lr * tm.schedule.get_lr(step)
        if cfg.optim == 'anvil':
            cfg.momentum = rail
        elif cfg.label == 'value_embeds':
            cfg.adam_betas,cfg.wd_mul = value_embed_betas_and_wd_mul(step)


def dense_step(tm, vgrad, step, table_event):
    configure_optimizer(tm,step)
    if table_event:
        tm.optimizer.adam_update_shard(tm.model.value_embeds,vgrad.mul_(1/8))
        vgrad.zero_()
    tm.optimizer.step(do_adam=step%2==1,sparse_update=None,deferred_labels=frozenset())
    if step == tm.split_step:
        tm.optimizer.copy_lm_state_to_embed()
        emit('embed_untied',step=step)


def row_plan(batches):
    pieces = []
    for b in batches:
        x = b['packed'].cuda(non_blocking=True)
        ids = torch.empty(2*x.numel(),dtype=torch.int32,device='cuda')
        hash_rows[(triton.cdiv(x.numel(),1024),)](x,ids,x.numel(),b['tokens'],1024)
        pieces.append(ids)
    ids = torch.cat(pieces)
    rows,inverse = torch.unique(ids,sorted=True,return_inverse=True)
    return rows,inverse


def cycles_for(total):
    from track_1_short.ngram_table import is_update_step
    cycles=[];current=[]
    for step in range(total):
        current.append(step)
        if is_update_step(step):
            cycles.append(current);current=[]
    if current:cycles.append(current)
    assert [s for c in cycles for s in c] == list(range(total))
    return cycles


def runner_key(schedule,step):
    from track_1_short.sampled_softmax import candidate_count_at
    index=next((i for i,(lo,hi) in enumerate(schedule.boundaries) if lo<=step<hi),4)
    return index,candidate_count_at(schedule,step)


def validation_batch(tokens, offset, T):
    from track_1_short.data import cu_seqlens_rows
    buf=tokens[offset:offset+T+1]
    assert buf.numel()==T+1
    x,y=buf[:-1].to(torch.int32),buf[1:].to(torch.int64)
    ends=torch.nonzero(x==50256)[:,0]
    cu=torch.full((cu_seqlens_rows(T),),T,dtype=torch.int32)
    cu[0]=0
    assert ends.numel()<cu.numel()
    cu[1:ends.numel()+1]=ends
    return x.pin_memory(),y.pin_memory(),cu.pin_memory()


@torch.no_grad()
def validate(model,tm,store,val_path,total_tokens,out,initial=False):
    from track_1_short.data import _load_data_shard
    from gh200_trace_plan import plan
    T=262144
    assert total_tokens%T==0
    tokens=_load_data_shard(val_path)
    losses=[];model.eval()
    for i,offset in enumerate(range(0,total_tokens,T)):
        x,y,cu=validation_batch(tokens,offset,T)
        rows,inv,_=plan(x,T)
        copy_rows(store,model.ngram_cache[:rows.numel()],rows)
        val=model(x.cuda(non_blocking=True),y.cuda(non_blocking=True),cu.cuda(non_blocking=True),
                  inv.to(torch.int32),tm.get_forward_args()).mean()
        number=float(val)
        assert np.isfinite(number),number
        losses.append(number)
        emit('initial_eval_batch' if initial else 'validation_batch',batch=i+1,total=total_tokens//T,loss=number)
        if not initial:write_json(out/'validation_progress.json',{'losses':losses,'tokens_per_batch':T})
    model.train()
    return float(np.mean(losses))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--data',type=Path,default=Path('/home/ubuntu/optimizer-research-cache/fineweb10B'))
    ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--limit-steps',type=int,default=1194,help='Only 1194 is a full run; shorter values are diagnostic.')
    ap.add_argument('--val-tokens',type=int,default=10485760)
    a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(8);torch.manual_seed(a.seed);torch.cuda.set_device(0)
    torch._dynamo.config.recompile_limit=64
    available=int(next(x.split()[1] for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))*1024
    assert available>200*2**30
    compat=install_attention_compat()
    from track_1_short.model.gpt import GPT
    from track_1_short.model.prefix_prediction import build_prefix_table_bucket
    from track_1_short.config import TRAINING_STAGES,LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT
    from track_1_short.schedule import TrainingSchedule
    from track_1_short.training import TrainingManager
    from track_1_short.tail_average import TailAverages
    from track_1_short.canonical_mask import BackgroundCanonicalMask
    from track_1_short.ngram_table import is_update_step,adam_beta2_and_wd_mul,NGRAM_LR_MUL
    from track_1_short.perf.kernels.ngram_adam import adam_rows_
    from track_1_short.perf.kernels.mlp import prime_stage_cache
    from track_1_short.perf.cuda_graphs.optimizer_graphs import AnvilBankGraphs
    dist.init_process_group('nccl',init_method='file://'+str((a.out/'dist-init').resolve()),rank=0,world_size=1,device_id=torch.device('cuda:0'))
    schedule=TrainingSchedule(TRAINING_STAGES,1122,20,torch.device('cuda'),LR_COOLDOWN_FRAC,SPLIT_EMBED_STAGE,WS_POST_YARN_EXT)
    assert 0<a.limit_steps<=schedule.total_steps
    full=a.limit_steps==schedule.total_steps
    model=GPT(50257,11,6,128,768,262144,ngram_dim=NGRAM_DIM,world_size=8,device=torch.device('cuda')).cuda()
    model.cast_matrix_weights_bf16();model.train()
    model.ngram_cache=torch.zeros(2*max(s.batch_size for s in TRAINING_STAGES)*4,NGRAM_DIM,dtype=torch.bfloat16,device='cuda')
    model.limit_yarn_rebuild(49152)
    initial={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    prefix=build_prefix_table_bucket(model.vocab_size,0,1)
    model.prefix_table.copy_(prefix)
    og=AnvilBankGraphs(torch.device('cuda'))
    tm=TrainingManager(model,schedule,bank_update=og.update)
    tm.advance_schedule(0)
    tail=TailAverages(tm.optimizer,0,is_update_step,schedule.total_steps)
    vgrad=torch.zeros_like(model.value_embeds,dtype=torch.float16)
    emit('allocate_cpu_table',bytes=NGRAM_VOCAB_SIZE*NGRAM_DIM*2)
    store=host_zeros((NGRAM_VOCAB_SIZE,NGRAM_DIM),torch.bfloat16)
    sv=host_zeros((NGRAM_VOCAB_SIZE,),torch.float32)
    sl=host_zeros((NGRAM_VOCAB_SIZE,),torch.int32)
    history=torch.zeros(schedule.total_steps+1,device='cuda')
    def check_placement():
        lines=numa_for(store)
        assert lines and all('bind:0' in x and not any(f'N{j}=' in x for j in range(1,9)) for x in lines),lines
    check_placement()
    result={'scope':__doc__,'status':'warming','seed':a.seed,'full_schedule':full,'steps_requested':a.limit_steps,
            'upstream_commit':'4ea6b937337a4889b8cfe3f38a93d120048d8f71','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'attention':compat,'stage_boundaries':schedule.boundaries,'val_tokens_requested':a.val_tokens,
            'table_bytes':store.nbytes,'timing':{},'stage_steps':[],'deviations':[
            'Descendant source, not certified record source; canonical mask follows corrected descendant',
            'Padded stock ARM CUDA12 FA3; floating-point reduction order differs',
            'Eight serial microbatches, explicit gradient averaging, shared delayed FP8 amax across logical ranks',
            'FP32 ngram gradient aggregation then BF16 cast; not official FP16/local and BF16/distributed merge',
            'CPU candidate/packing lookahead is one background thread; not original eight-rank pipeline',
            'Evaluation compute reported separately and in full wall clock; not an official leaderboard timer']}
    write_json(a.out/'results.json',result)
    # Check the full-sized BF16 evaluation path before investing in training.
    t=time.perf_counter();model.complete_yarn_tables()
    result['initial_validation_loss']=validate(model,tm,store,a.data/'fineweb_val_000000.bin',262144,a.out,initial=True)
    result['timing']['initial_eval_preflight_seconds']=time.perf_counter()-t
    prime_stage_cache()
    model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True)
    warm=DataStream(a.data,schedule,prefix)
    pool=torch.cuda.graph_pool_handle();runners={};seen=set();warmsteps=[]
    for step in range(schedule.total_steps):
        key=runner_key(schedule,step)
        if key not in seen:seen.add(key);warmsteps.append(step)
    for step in warmsteps:
        t=time.perf_counter();tm.advance_schedule(step)
        batch=warm.next(step)
        model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True)
        runner=Runner(model,tm,batch,vgrad,pool)
        emit('capture_begin',step=step,key=runner_key(schedule,step))
        runner.capture();runners[runner_key(schedule,step)]=runner
        # Exercise both ordinary Adam and the value-embedding update, then restore all state below.
        runner.forward_backward()
        dense_step(tm,vgrad,step|1,True)
        torch.cuda.synchronize()
        emit('capture_complete',step=step,seconds=time.perf_counter()-t,allocated_gib=torch.cuda.memory_allocated()/2**30)
    # Warm the untied embedding Adam path; all learned state is restored after optimizer graph capture.
    tm.optimizer.copy_lm_state_to_embed()
    for p,cfg in tm.optimizer.param_cfgs.items():
        if cfg.label=='embed':p.grad=torch.zeros_like(p)
    dense_step(tm,vgrad,1193,False)
    configure_optimizer(tm,0);tm.optimizer.stage_bank_scalars()
    for bank in tm.optimizer.banks.values():bank.grad.zero_()
    og.capture([tm.optimizer.banks[k] for k in ['qk_bank','vo_bank','mlp_bank']])
    for bank in tm.optimizer.banks.values():og.update(bank)
    og.seal()
    model.load_state_dict(initial);del initial
    for p,state in tm.optimizer.param_states.items():
        for key,value in state.items():
            if torch.is_tensor(value):value.zero_()
            elif key=='step':state[key]=0
        p.grad=None
    tm.reset();tm.advance_schedule(0);vgrad.zero_();model.ngram_cache.zero_()
    model.rearm_fp8_bootstrap();model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=True)
    warm.close();del warm,batch,runner
    canon=BackgroundCanonicalMask(model.vocab_size,True,lambda s,**kw:emit('canonical',message=s))
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    result['timing']['startup_and_warmup_seconds']=time.perf_counter()-PROCESS_START
    result['status']='training';write_json(a.out/'results.json',result)
    emit('training_start',startup_seconds=result['timing']['startup_and_warmup_seconds'])
    train_start=time.perf_counter();canon.start()
    stream=DataStream(a.data,schedule,prefix)
    cycles=cycles_for(a.limit_steps);executor=ThreadPoolExecutor(max_workers=1)
    future=executor.submit(stream.cycle,cycles[0]);events=0;stage_times={};stage_counts={}
    try:
        for ci,steps in enumerate(cycles):
            cycle_start=time.perf_counter();wait_start=time.perf_counter();batches=future.result();wait_seconds=time.perf_counter()-wait_start
            future=executor.submit(stream.cycle,cycles[ci+1]) if ci+1<len(cycles) else None
            rows,inv=row_plan(batches);n=rows.numel();assert n<=model.ngram_cache.shape[0]
            cache=model.ngram_cache[:n]
            v=torch.empty(n,device='cuda');last=torch.empty(n,dtype=torch.int32,device='cuda');local=torch.arange(n,dtype=torch.int32,device='cuda')
            acc=torch.zeros(n,NGRAM_DIM,device='cuda')
            copy_rows(store,cache,rows);copy_state(sv,sl,v,last,rows)
            pointer=0
            for b in batches:
                step=b['step'];tm.advance_schedule(step);key=runner_key(schedule,step);runner=runners[key]
                start=time.perf_counter()
                model.quantize_attn_fp8();model.quantize_mlp_fp8(refresh_lm=step%2==0)
                losses=[]
                for micro in b['micros']:
                    slots=inv[pointer:pointer+2*b['tokens']].to(torch.int32);pointer+=2*b['tokens']
                    runner.stage(micro,slots);g=runner.forward_backward()
                    accumulate_rows[(triton.cdiv(g.numel(),4096),)](g,slots,acc,slots.numel(),NGRAM_DIM,4096,num_warps=4)
                    losses.append(runner.l.detach().clone()/b['tokens'])
                event=is_update_step(step)
                if event:
                    assert step==steps[-1]
                    events+=1;beta,wd=adam_beta2_and_wd_mul(step);history[events]=beta
                    lr=.008*schedule.get_lr(step)*NGRAM_LR_MUL
                    adam_rows_(cache,v,last,history,acc.to(torch.bfloat16),local,local,events,beta2=beta,eps=1e-10,
                               step_size=lr*(1-beta**events)**.5,decay=lr*lr*.005*wd,grad_mul=1/8,sq_mul=(1-beta)/64)
                    copy_rows(cache,store,rows,True);copy_state(v,last,sv,sl,rows,True)
                dense_step(tm,vgrad,step,event);tail.tick(step)
                torch.cuda.synchronize();seconds=time.perf_counter()-start
                objective=float(torch.stack(losses).mean());assert np.isfinite(objective),('nonfinite objective',step,objective)
                stage_times[key[0]]=stage_times.get(key[0],0)+seconds;stage_counts[key[0]]=stage_counts.get(key[0],0)+1
                with (a.out/'steps.jsonl').open('a') as f:f.write(json.dumps({'step':step+1,'stage':key[0],'global_tokens':8*b['tokens'],'seconds_excluding_cycle_pull':seconds,'training_objective':objective})+'\n')
                if (step+1)%25==0 or step+1==a.limit_steps:
                    emit('step',step=step+1,total=a.limit_steps,train_seconds=time.perf_counter()-train_start,objective=objective)
                    result['last_step']=step+1;result['status']='training';write_json(a.out/'results.json',result)
            assert pointer==inv.numel();check_placement()
            with (a.out/'cycles.jsonl').open('a') as f:f.write(json.dumps({'first_step':steps[0],'last_step':steps[-1],'seconds':time.perf_counter()-cycle_start,'data_wait_seconds':wait_seconds,'unique_rows':n,'table_event':is_update_step(steps[-1])})+'\n')
            del batches,rows,inv,cache,v,last,local,acc,g,slots,losses
    finally:
        executor.shutdown(wait=True);stream.close()
    torch.cuda.synchronize()
    result['timing']['training_loop_seconds']=time.perf_counter()-train_start
    result['training_shards']=stream.loaded
    result['stage_steps']=[{'stage':i,'steps':stage_counts[i],'seconds_excluding_cycle_pull':stage_times[i]} for i in sorted(stage_counts)]
    result['table_update_events']=events;result['status']='finalizing';write_json(a.out/'results.json',result)
    emit('training_complete',seconds=result['timing']['training_loop_seconds'],steps=a.limit_steps)
    t=time.perf_counter();canon.wait();canon.collect(model.canon_mask)
    if full:
        tail.ship();tm.advance_schedule(schedule.total_steps);tm.apply_final_ws_ext()
    model.complete_yarn_tables();torch.cuda.synchronize()
    result['timing']['tail_mask_finalize_seconds']=time.perf_counter()-t
    # Free all training graph pools before the full-size BF16 validation.
    del runner,runners,og,tail,vgrad,tm.optimizer
    gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
    result['status']='validating';write_json(a.out/'results.json',result)
    t=time.perf_counter()
    result['validation_loss']=validate(model,tm,store,a.data/'fineweb_val_000000.bin',a.val_tokens,a.out)
    result['timing']['validation_seconds']=time.perf_counter()-t
    result['timing']['process_seconds_through_validation']=time.perf_counter()-PROCESS_START
    result['gpu_peak_allocated_bytes']=torch.cuda.max_memory_allocated();result['gpu_peak_reserved_bytes']=torch.cuda.max_memory_reserved()
    result['status']='completed';result['completed']=True
    check_placement();result['table_cpu_only_verified']=True
    write_json(a.out/'results.json',result)
    emit('completed',loss=result['validation_loss'],timing=result['timing'],full_schedule=full)
    dist.destroy_process_group()


if __name__=='__main__':
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
