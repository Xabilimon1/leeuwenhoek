# Exploratory: mirror-symmetric initialisation (2026-11-24, NOT pre-registered)

Quick scripts from a brainstorm (Fable agent, checked by Claude). Short proxy
training: AlgZoo's recipe compressed to 2^26 sequences (~20 s per net). Post hoc;
nothing here is a result until the paired, pre-registered full-recipe experiment.

Run from the repo root: `PYTHONPATH=src .venv/bin/python scripts/explore_mirror_init/b1.py scripts/explore_mirror_init <mode> <seeds...>`

- `tr.py`: proxy trainer (`train`), mirror copy of one half of the net into the other (`mirror_copy`), `report` (exact accuracy, mirror residual, near-tie share, agreement with ARC seed 0 up to x -> -x, pilot criterion).
- `b1.py`: mode `sym` = full mirror init, any other mode = normal init.
- `b2.py`: control, only w_ih mirrored.
- `b3.py <t> <seeds>`: init = t * mirror + (1 - t) * normal.
- `a.py`: init statistics of the 21 local nets vs final accuracy (needs scipy, absent from .venv).
- `split_test.py`: init seed vs data seed split by re-seeding after the step-0 save; 5 checks, all True on 2026-11-24 (CPU training bit-identical across reruns).

What was seen (Fable, seeds 5-19): full mirror init 10/15 pass the pilot criterion vs 2/15 with the full normal recipe; w_ih-only mirror 0/10; mix t = 0.9: 3/5, t = 0.5: 0/5 (on seeds that had worked). Reproduced by Claude on seeds 5-12: pass 6, 7, 8, 11 (4/8), fail 5, 9, 10, 12, identical numbers. Caveats: seeds 6 and 8 agree with ARC seed 0 only 96.9 / 97.6 % (another symmetric solution, or proxy too short); seed 9 has the mechanism with normal init but loses it with mirror init.
