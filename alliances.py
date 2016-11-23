from collections import defaultdict
from tabulate import tabulate
import numpy as np
print "Results after 1 throw"

table = []

MAXIMUM_DEFENDER_INFANTRY = 6
MAXIMUM_ATTACKER_INFANTRY = 6
for di in xrange(1,MAXIMUM_DEFENDER_INFANTRY+1):
    row = list()
    row.append("%d+%d"%(di,0))
    for ai in xrange(1,MAXIMUM_ATTACKER_INFANTRY+1):
        results = defaultdict(list)
        for i in xrange(10000):
            dthrow = np.random.randint(1,7,size=di)
            athrow = np.random.randint(1,7,size=ai)

            results["d_wounded"].append(sum((3 <= athrow) * (athrow < 5)))
            results["d_killed"].append(sum(5 <= athrow))
            results["d_healthy"].append(di - results["d_killed"][-1] - results["d_wounded"][-1] )

            results["a_wounded"].append(sum(dthrow < 5))
            results["a_killed"].append(sum(5 <= dthrow))
            results["a_healthy"].append(ai - results["a_killed"][-1] - results["a_wounded"][-1] )

            if results["a_wounded"][-1] + results["a_healthy"][-1] > 0 and results["d_healthy"][-1]<=0:
                results["won"].append(True)
            else:
                results["won"].append(False)

        row.append(np.mean(np.array(results["won"], dtype='float32')))
    table.append(row)

print tabulate(table, ["%d+0"%i for i in xrange(1,MAXIMUM_ATTACKER_INFANTRY+1)], tablefmt="fancy_grid")