import numpy as np, torch as th, time
from itertools import permutations
from pathlib import Path
from alg_zoo.architectures import DistRNN
from leeuwenhoek.models import load_argmedian, numpy_weights
from leeuwenhoek.exact import exact_accuracy
from leeuwenhoek.structure import mirror_permutation
from leeuwenhoek.forward import forward
D=Path("data/trained")
def ld(seed,n):
    m=DistRNN(hidden_size=4,seq_len=3); m.load_state_dict(th.load(D/f"argmedian_4_3_seed{seed}/argmedian_4_3_{n}_{seed}.pth",weights_only=True)); return numpy_weights(m)
t=time.time()
rows=[]
for s in [0]+list(range(5,25)):
    w0=ld(s,0); wf=ld(s,2**30)
    acc=exact_accuracy(wf)[0]; res=mirror_permutation(wf)[1]; r0=mirror_permutation(w0)[1]
    sg0=''.join('+' if v>0 else '-' for v in w0['w_ih']); sgf=''.join('+' if v>0 else '-' for v in wf['w_ih'])
    rows.append((s,acc,res,r0,sg0,sgf,np.abs(w0['w_ih']).round(2), np.linalg.norm(w0['w_hh']), np.linalg.norm(w0['w_out'])))
    print(s,f"{acc:.4f} res={res:.3f} res0={r0:.3f} {sg0}->{sgf} |wih0|={np.abs(w0['w_ih']).round(2)} hh0={np.linalg.norm(w0['w_hh']):.2f} out0={np.linalg.norm(w0['w_out']):.2f}")
print("t",time.time()-t)
import collections
c=collections.defaultdict(list)
for r in rows: c[sum(ch=='+' for ch in r[5])].append(round(r[1]*100,1))
print("final #pos w_ih -> accs",dict(c))
c=collections.defaultdict(list)
for r in rows: c[sum(ch=='+' for ch in r[4])].append(round(r[1]*100,1))
print("init #pos w_ih -> accs",dict(c))
acc=np.array([r[1] for r in rows])
for name,v in [("res0",[r[3] for r in rows]),("hh0",[r[7] for r in rows]),("out0",[r[8] for r in rows]),("min|wih0|",[r[6].min() for r in rows]),("sum|wih0|",[r[6].sum() for r in rows])]:
    from scipy.stats import spearmanr
    print(name,"spearman vs acc",spearmanr(v,acc))
