import sys, time, functools, numpy as np, torch as th
from alg_zoo.architectures import DistRNN
from alg_zoo.tasks import task_registry
from alg_zoo.training import lr_mult_schedule
from leeuwenhoek.models import load_argmedian, numpy_weights
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.structure import mirror_permutation
from leeuwenhoek.forward import forward
th.set_num_threads(1)
task=task_registry["argmedian"][0]
def mirror_copy(model):
    sd=model.state_dict(); P=[3,2,1,0]
    wi=sd["rnn.weight_ih_l0"]; wh=sd["rnn.weight_hh_l0"]; wo=sd["linear.weight"]
    with th.no_grad():
        for i in (0,1):
            wi[P[i],0]=-wi[i,0]
            for j in range(4): wh[P[i],P[j]]=wh[i,j]
            wo[:,P[i]]=wo[:,i]
def scale(model,f_ih=1,f_hh=1,f_out=1):
    sd=model.state_dict()
    with th.no_grad():
        sd["rnn.weight_ih_l0"]*=f_ih; sd["rnn.weight_hh_l0"]*=f_hh; sd["linear.weight"]*=f_out
def train(seed, n_sched=2**26, stop=None, init=None, data_seed=None, lr_base=0.04, bs=1024, wd=0.001, warm=2**14):
    th.manual_seed(seed)
    model=DistRNN(hidden_size=4,seq_len=3)
    if init: init(model)
    if data_seed is not None: th.manual_seed(data_seed)
    opt=th.optim.AdamW(model.parameters(),lr=lr_base/2,weight_decay=wd)
    n=n_sched//bs
    sch=th.optim.lr_scheduler.LambdaLR(opt,functools.partial(lr_mult_schedule,n_steps=n,warmup_steps=warm,decay_min_mult=0.01))
    for step in range((stop or n_sched)//bs):
        x=th.randn(bs,3); loss=th.nn.functional.cross_entropy(model(x),task(x))
        loss.backward(); opt.step(); sch.step(); opt.zero_grad()
    return numpy_weights(model)
rng=np.random.default_rng(1); X=rng.standard_normal((300000,3)).astype(np.float32)
gap=np.diff(np.sort(X,-1),axis=-1).min(-1)/np.abs(X).max(-1)
tgt=np.argsort(X,-1)[:,1]
_,w_arc=load_argmedian(4,3,seed=0); a0=forward(w_arc,X).argmax(-1); a0m=forward(w_arc,-X).argmax(-1)
def report(tag,w):
    acc=exact_accuracy(w)[0]; res=mirror_permutation(w)[1]; p=forward(w,X).argmax(-1); wrong=p!=tgt
    near=np.mean(gap[wrong]<0.05); agree=max(np.mean(p==a0),np.mean(p==a0m))
    mech = res<0.1 and near>=0.8
    print(f"{tag}: acc={acc:.4%} mirror={res:.3f} near-tie={near:.0%} agree_ARC0={agree:.2%} MECH={mech} w_ih={np.round(w['w_ih'],2)}",flush=True)
    return mech
