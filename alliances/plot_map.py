
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState
from scipy.spatial import Voronoi, voronoi_plot_2d

import cma
from alliances_map import borders, names
from scipy.spatial import Delaunay
import networkx as nx

N = len(borders)

rng = RandomState(317071)
points = np.array([rng.randn(2) for node in borders])  # start with a random initialization
edge_list = [(i,b)  for i,neighbor in enumerate(borders) for b in neighbor]
if True:  # springy optimization
    G = nx.Graph()
    G.add_nodes_from(range(len(borders)))
    G.add_edges_from(edge_list)
    springy_result = nx.spring_layout(G, dim=2,iterations=10000)

    points = np.array([springy_result[i] for i in range(len(borders))])

if True:  # cmaes optimization
    def iter_test_safe(params):
        """
        Test if a certain set of coordinates is OK.
        Return an error if not
        :param param:
        :return:
        """
        points = params.reshape(-1,2)
        tri = Delaunay(points)
        indices, indptr = tri.vertex_neighbor_vertices
        neighbour_graph = [indptr[indices[k]:indices[k+1]] for k in range(len(indices)-1)]
        error = 0
        for neighbours, target_neighbours in zip(neighbour_graph, borders):
            for neighbour in neighbours:
                if neighbour not in target_neighbours:
                    error += 1
            for target_neighbour in target_neighbours:
                if target_neighbour not in neighbours:
                    error += 1
        return error/2

    options = {'ftarget':0, 'seed':1}

    result = cma.fmin(iter_test_safe, points.flatten(), sigma0=0.2, restarts=0, options=options)[0]
    print result
    points = result.reshape(-1,2)

# compute Voronoi tesselation
vor = Voronoi(points)

# plot
voronoi_plot_2d(vor)

# colorize
for region in vor.regions:
    if not -1 in region:
        polygon = [vor.vertices[i] for i in region]
        plt.fill(*zip(*polygon))

plt.figure()
for A,B in vor.ridge_points:
    if A in borders[B]:
        plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'g-', lw=2)
    else:
        plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'r-', lw=2)

for edge in edge_list:
    if edge not in vor.ridge_points and edge[::-1] not in vor.ridge_points:
        A,B = edge
        plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'b-', lw=2)

for i,name in enumerate(names):
    plt.text(points[i][0], points[i][1], name, verticalalignment='center', horizontalalignment='center',size=8,
        bbox={'facecolor':'white', 'alpha':0.7, 'pad':0})

plt.show(block=True)