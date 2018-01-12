#!/usr/bin/env python

import numpy as np
import sys


N = int(sys.argv[1])
print N

A = np.random.rand(N, N) * 10.0
A = np.dot(A, A.T)

f = file('A%sx%s.txt' % (N, N), 'w')
f.write('%s %s\n' % (N, N))
for j in range(N):
    for i in range(N):
        if i == j:
            A[i, j] += 1
        f.write('%s ' % A[i, j])
    f.write('\n')
f.close()

b = np.dot(A, np.ones(N))

f = file('b%sx%s.txt' % (1, N), 'w')
f.write('%s %s\n' % (1, N))
for i in range(N):
    f.write('%s ' % b[i])
f.write('\n')
f.close()
