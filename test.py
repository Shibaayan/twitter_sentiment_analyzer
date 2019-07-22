# -*- coding: utf-8 -*-
from tqdm import tqdm

x = 1

for i in tqdm(range(0,20000)):
    for x in range(0,10000):
        x *= 4