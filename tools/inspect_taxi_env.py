import gymnasium as gym
import inspect, sys, pprint

print('Creating env...')
env = gym.make('Taxi-v3')
inner = env.unwrapped
print('\nInner env class: ', inner.__class__)
print('Module: ', inner.__class__.__module__)
try:
    f = inspect.getsourcefile(inner.__class__)
    print('Source file: ', f)
except Exception as e:
    print('getsourcefile failed:', e)

print('\nSelected public attributes:')
attrs = [a for a in dir(inner) if not a.startswith('_')]
print(attrs)

print('\n__dict__ small primitive items:')
d = getattr(inner, '__dict__', {})
for k, v in d.items():
    try:
        if isinstance(v, (int, float, str, bool)):
            print(k, ':', v)
        elif isinstance(v, (list, tuple)) and len(v) <= 10 and all(isinstance(x, (int, float, str, bool)) for x in v):
            print(k, ':', v)
    except Exception:
        pass

print('\nHas attribute s? ', hasattr(inner, 's'))
print('Has decode? ', hasattr(inner, 'decode'))
print('Has encode? ', hasattr(inner, 'encode'))
print('Other candidate names:')
print([name for name in dir(inner) if any(k in name.lower() for k in ('state', 's', 'encode', 'decode', 'restore', 'set'))])

# Try to print first 200 lines of class source if available
try:
    src = inspect.getsource(inner.__class__)
    print('\n--- class source (first 200 lines) ---')
    lines = src.splitlines()
    for ln in lines[:200]:
        print(ln)
except Exception as e:
    print('getsource failed:', e)

print('\nDone')
