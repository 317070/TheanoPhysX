
import numpy as np
import matplotlib.pyplot as plt
from numpy.random.mtrand import RandomState
from scipy.spatial import Voronoi, voronoi_plot_2d

import cma
from alliances_map import borders, names
from scipy.spatial import Delaunay
import networkx as nx
import itertools as it
from networkx.algorithms import bipartite
import math

# names = ['A','B','C']
# borders =[[1,2],[0,2],[0,1]]
N = len(borders)
# names = range(N)
rng = RandomState(317071)
points = np.array([rng.randn(2) for node in borders])  # start with a random initialization
edge_list = [(i,b)  for i,neighbor in enumerate(borders) for b in neighbor]
single_edge_list = [edge for edge in edge_list if edge[0]<=edge[1]]

def is_planar(G):
    """
    function checks if graph G has K(5) or K(3,3) as minors,
    returns True /False on planarity and nodes of "bad_minor"
    """
    result=True
    bad_minor=[]
    n=len(G.nodes())
    if n>5:
        for subnodes in it.combinations(G.nodes(),6):
            subG=G.subgraph(subnodes)
            if subG.number_of_edges()>=9:
                print "OI"
                if bipartite.is_bipartite(subG):# check if the graph G has a subgraph K(3,3)
                    X, Y = bipartite.sets(subG)
                    if len(X)==3:
                        result=False
                        bad_minor=(X.pop(),X.pop(),X.pop(),Y.pop(),Y.pop(),Y.pop())
                        print [(names[i], names[j]) for i,j in subG.edges()]
                        return result,bad_minor
    if n>4 and result:
        for subnodes in it.combinations(G.nodes(),5):
            subG=G.subgraph(subnodes)
            if len(subG.edges())==10:# check if the graph G has a subgraph K(5)
                result=False
                bad_minor=subnodes
                return result,bad_minor
    return result,bad_minor

def find_planar_subgraph(G):
    if len(G)<3:
        return G
    else:
        is_planar_boolean, bad_minor = is_planar(G)
        if is_planar_boolean:
            return G
        else:
            print "BAD!", bad_minor
            for i in bad_minor:
                print names[i]
            for i in xrange(3):
                for j in xrange(3):
                    if (i,j) in edge_list:
                        print True
            G.remove_node(bad_minor[0])
            return find_planar_subgraph(G)


# simplify until the graph is planar!
# Create 4-point-crossing where they cross
G = nx.Graph()
G.add_nodes_from(range(i))
G.add_edges_from(single_edge_list)

G_fixed = find_planar_subgraph(G)






# make a planar embedding
# step 1: find a triangle
def get_triangle():
    for edge1 in edge_list:
        for edge2 in edge_list:
            if edge1[1]==edge2[0]:
                if (edge2[1],edge1[0]) in edge_list:
                    return [edge1[0], edge2[0], edge2[1]]

triangle = get_triangle()
print triangle

# make a big system out of it!
A = np.zeros(shape=(2*N,2*N))
B = np.zeros(shape=(2*N))

for i, neighbors in enumerate(borders):
    if i not in triangle:
        for n in neighbors:
            A[2*i,  2*n] = 1./len(neighbors)
            A[2*i+1,2*n+1] = 1./len(neighbors)
        A[2*i,  2*i] = -1.
        A[2*i+1,2*i+1] = -1.
    else:
        idx = triangle.index(i)
        tr = [(-1,-1),(0,1),(1,-1)]
        A[2*i,  2*i] = 1.
        A[2*i+1,2*i+1] = 1.
        B[2*i] = tr[idx][0]
        B[2*i+1] = tr[idx][1]

x = np.linalg.solve(A,B)

points = x.reshape(-1,2)



# have faces with edges
edges_to_do = list(edge_list)
faces = []

def clockwise_angle(edge1, edge2):
    angle1 = math.atan2(points[edge1[1]][1] - points[edge1[0]][1], points[edge1[1]][0] - points[edge1[0]][0])
    angle2 = math.atan2(points[edge2[1]][1] - points[edge2[0]][1], points[edge2[1]][0] - points[edge2[0]][0])

    res = np.rad2deg((angle1 - angle2 + math.pi) % (2 * math.pi))
    if res==0:
        res+=360
    return res


while edges_to_do:
    todo = edges_to_do.pop()
    face = [todo]
    while face[0][0] != face[-1][1]:
        min_angle = 361
        min_edge = None
        for edge in edges_to_do:
            if edge[0]==face[-1][1]:
                ang = clockwise_angle(face[-1], edge)
                if ang<min_angle:
                    min_edge = edge
                    min_angle = ang
        face.append(min_edge)
        edges_to_do.remove(min_edge)
    faces.append(face)




# create dual graph
dual_graph = []


if False:  # springy optimization
    G = nx.Graph()
    G.add_nodes_from(range(len(borders)))
    G.add_edges_from(edge_list)
    springy_result = nx.spring_layout(G, dim=2,iterations=10000)
    # springy_result = nx.spectral_layout(G, dim=2)
    points = np.array([springy_result[i] for i in range(len(borders))])


if False:  # cmaes optimization
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
                    error += 2
            for target_neighbour in target_neighbours:
                if target_neighbour not in neighbours:
                    error += 0
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
if False:
    for A,B in vor.ridge_points:
        if A in borders[B]:
            plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'g-', lw=2)
        else:
            plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'r-', lw=2)

    ridge_points =[(a,b) for a,b in vor.ridge_points]

    for edge in edge_list:
        if edge not in ridge_points and edge[::-1] not in ridge_points:
            A,B = edge
            plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'b-', lw=1)

for edge in edge_list:
    A,B = edge
    plt.plot([points[A][0], points[B][0]], [points[A][1], points[B][1]], 'b-', lw=1)


for i,name in enumerate(names):
    plt.text(points[i][0], points[i][1], name, verticalalignment='center', horizontalalignment='center',size=8,
        bbox={'facecolor':'white', 'alpha':0.7, 'pad':0})

plt.show(block=True)