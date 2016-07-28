import numpy as np
import scipy.linalg

import json

r = []

for object, reference in [("spine", None),
                          ("tibia1", "spine"),
                          ("tibia2", "spine"),
                          ("tibia3", "spine"),
                          ("tibia4", "spine"),
                          ("femur1", "tibia1"),
                          ("femur2", "tibia2"),
                          ("femur3", "tibia3"),
                          ("femur4", "tibia4"),]:

    for i in xrange(3):
        a = [0,0,0]
        a[i] = 1
        for type in ["velocity", "position", "orientation"]:
            d = {
              "type": type,
              "object": object,
              "axis": a
            }
            if reference:
                d["reference"] = reference
            r.append(d)
print len(r)
print json.dumps(r, sort_keys=True, indent=2)