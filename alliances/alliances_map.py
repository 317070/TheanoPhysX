import json

BORDERS = "borders"
RECRUITS = "recruit"
INVESTMENTS = "invest"

json_data = open("alliance_map.json","r").read()
mapdict = json.loads(json_data)

for region, region_dict in mapdict.iteritems():
    for neighbor in region_dict[BORDERS]:
        if neighbor not in mapdict:
            print "%s bordering %s is unknown"%(neighbor, region)
        elif region not in mapdict[neighbor][BORDERS]:
            print "%s bordering %s is not reciprocal"%(neighbor, region)

name_to_id = {region:i for i,region in enumerate(mapdict)}
regions = [region for region in mapdict]
borders = [[name_to_id[b] for b in mapdict[region][BORDERS]] for region in mapdict]
recruits = [region_data[RECRUITS] for region_data in mapdict.itervalues()]
maximum_investments = [region_data[INVESTMENTS] for region_data in mapdict.itervalues()]
names = mapdict.keys()
# TODO: pareto plot
# What is the maximum number of soldiers, you can pay and have as soon as possible
# What is the fastest snowball you can do?

# How many turns does it take minimally to get X paid units on the board?
# def minimum_turn_to_X_paid_units(X, **kwargs):
#     units = [0 for i in xrange(len(regions))]
#     for k,v in kwargs.iteritems():
#         k = k.replace("_"," ")
#         units[name_to_id[k]] = v
#     print units
#
# minimum_turn_to_X_paid_units(15, West_Germany=1, East_Germany=2)
