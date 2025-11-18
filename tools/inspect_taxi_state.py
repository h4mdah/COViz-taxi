import gymnasium as gym
import inspect, pprint

env = gym.make('Taxi-v3')
res = env.reset()
print('reset returned (type):', type(res))
if isinstance(res, tuple):
    print('obs:', res[0])
else:
    print('obs:', res)
inner = env.unwrapped
print('\ninner class:', inner.__class__)
print('Module:', inner.__class__.__module__)

print('\n__dict__ keys and primitive values AFTER reset:')
for k, v in inner.__dict__.items():
    if isinstance(v, (int, float, str, bool)):
        print(k, ':', v)
    else:
        print(k, ':', type(v))

print('\nHas attribute s? ', hasattr(inner, 's'))
print('getattr state: ', getattr(inner, 'state', None))
print('Has decode? ', hasattr(inner, 'decode'))
print('Has encode? ', hasattr(inner, 'encode'))

# search source for occurrences of 'self.s' or 's =' or 'self._s'
src = inspect.getsource(inner.__class__)
print('\nLines mentioning self.s or self._s or self.state:')
for i, line in enumerate(src.splitlines()):
    if 'self.s' in line or 'self._s' in line or 'self.state' in line or ' self =' in line:
        print(i+1, line)

print('\n--- reset method source ---')
for name, func in inspect.getmembers(inner.__class__, predicate=inspect.isfunction):
    if name == 'reset' or name == 'step':
        print(f'-- {name} --')
        try:
            print(inspect.getsource(func))
        except Exception as e:
            print('could not get source for', name, e)

print('\ndone')
