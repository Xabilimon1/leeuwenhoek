import sys, copy; sys.path.insert(0,sys.argv[1]); from tr import *
t_mix=float(sys.argv[2]); seeds=[int(s) for s in sys.argv[3:]]
def mix(model):
    orig={k:v.clone() for k,v in model.state_dict().items()}
    mirror_copy(model)
    sd=model.state_dict()
    with th.no_grad():
        for k in sd: sd[k].copy_(t_mix*sd[k]+(1-t_mix)*orig[k])
t=time.time()
for s in seeds:
    report(f"t={t_mix} seed {s}", train(s, init=mix))
print("time",time.time()-t)
