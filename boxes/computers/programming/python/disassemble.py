import dis
from pprint import pprint
import time


def count_up_and_wait():
    print("Hello World")
    for i in range(100):
        input()
        print(i)


d = {}
dd = dict()

print(dis.dis(count_up_and_wait))

start = time.now()
print(dis.dis("{'key': 'value'}"))
print(time.now() - start)

start = time.now()
print(dis.dis("dict(key='value')"))
print(time.now() - start)
