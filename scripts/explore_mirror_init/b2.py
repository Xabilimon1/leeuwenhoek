import sys; sys.path.insert(0,sys.argv[1]); from tr import *
def wih_only(model):
    wi=model.state_dict()["rnn.weight_ih_l0"]
    with th.no_grad(): wi[3,0]=-wi[0,0]; wi[2,0]=-wi[1,0]
mode=sys.argv[2]; seeds=[int(s) for s in sys.argv[3:]]
t=time.time()
for s in seeds:
    report(f"{mode} seed {s}", train(s, init=wih_only if mode=="wih" else None))
print("time",time.time()-t)
