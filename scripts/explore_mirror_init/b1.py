import sys; sys.path.insert(0,sys.argv[1]); from tr import *
mode=sys.argv[2]; seeds=[int(s) for s in sys.argv[3:]]
t=time.time()
for s in seeds:
    report(f"{mode} seed {s}", train(s, init=mirror_copy if mode=="sym" else None))
print("time",time.time()-t)
