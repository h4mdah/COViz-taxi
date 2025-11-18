from pathlib import Path
p = Path('results/side_by_side_env_regen')
if not p.exists():
    print('no outputs')
else:
    files = sorted(p.glob('*.gif'))
    print('found', len(files), 'gifs')
    for f in files:
        print(f)
