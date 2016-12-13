# coding=utf-8
from collections import defaultdict
from math import floor
import numpy as np
from texttable import Texttable

MAXIMUM_DEFENDER_INFANTRY = 6
MAXIMUM_ATTACKER_INFANTRY = 8

texttable = Texttable(max_width=0)

header = []
for ai in xrange(1,MAXIMUM_ATTACKER_INFANTRY+1):
    for aa in xrange(ai/2,-1,-1):
        header.append("%d+%d"%(ai-2*aa,aa))

ncols = len(header)
texttable.set_cols_align(["r"]+ncols*["c"])
texttable.set_cols_valign(["t"]+ncols*["b"])
texttable.add_row(["% won\nK/D"]+header)

for di in xrange(1,MAXIMUM_DEFENDER_INFANTRY+1):
    for trench in [False, True]:
        row = list()
        row.append("%d\n%s"%(di, "trench" if trench else ""))
        for ai in xrange(1,MAXIMUM_ATTACKER_INFANTRY+1):
            for aa in xrange(ai/2,-1,-1):
                att_infa = ai - 2*aa
                att_arti = aa
                def_unit = di
                results = defaultdict(list)

                for i in xrange(1000):
                    stop = False
                    d_killed, d_wounded, d_healthy = 0, 0, def_unit
                    a_killed, a_wounded, a_healthy = 0, 0, att_infa
                    art_killed, art_wounded, art_healthy = 0, 0, att_arti
                    while not stop:
                        d_killed += min(art_healthy, d_healthy)
                        d_healthy -= min(art_healthy, d_healthy)
                        if d_healthy<=0:
                            stop=True
                            break
                        dthrow = np.random.randint(1,7,size=d_healthy)
                        athrow = np.random.randint(1,7,size=a_healthy)

                        if not trench:
                            killed_now = min(sum(5 <= athrow), d_healthy)
                            d_killed += killed_now
                            d_wounded += min(sum((3 <= athrow) * (athrow < 5)), d_healthy - killed_now)
                        else:
                            d_killed += min(sum(5 <= athrow)/2, d_healthy)
                            d_wounded += 0

                        d_healthy = def_unit - d_killed - d_wounded

                        killed_now = min(sum(5 <= dthrow), att_infa+att_arti)
                        a_killed += killed_now
                        a_wounded += min(sum(dthrow < 5), att_infa+att_arti - killed_now)
                        # attacking side chooses artillery to die last
                        a_healthy = att_infa - a_killed - a_wounded

                        art_healthy = att_arti - max(-a_healthy, 0)
                        a_healthy = max(a_healthy, 0)

                        if a_healthy<=0 or d_healthy<=0:
                            stop=True
                        elif a_healthy<=d_healthy:
                            # ADD AI to stop when all is lost
                            stop=True
                        else:
                            pass

                    if a_healthy + art_healthy <= 0:
                        results["won"].append(False)
                    elif d_healthy<=0:
                        results["won"].append(True)
                    else:
                        results["won"].append(False)

                    results["d_killed"].append(d_killed)
                    results["d_wounded"].append(d_wounded)
                    results["a_killed"].append(a_killed)
                    results["a_wounded"].append(a_wounded)


                won = np.mean(np.array(results["won"], dtype='float32'))
                akill = np.mean(np.array(results["d_killed"], dtype='float32'))
                aloss = np.mean(np.array(results["a_killed"], dtype='float32'))
                if won>0:
                    row.append("%.3f\n%.1f/%.1f"%(won, akill, aloss))
                else:
                    if akill>2*aloss:
                        #row.append("")
                        row.append("\n%.1f/%.1f"%(akill, aloss))
                    else:
                        row.append("")
        texttable.add_row(row)

print texttable.draw()