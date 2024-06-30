
import pyserver
import numpy as np

s = pyserver.Server(3000)
def process(n):
    print(n)
    d = { "x": 1, "y": 1 }
    return d

s.process(process)
s.open_window()

