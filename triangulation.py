import numpy as np
import triangle as tr


# returns a list of segments (pt_ini, pt_fin) from a list of coordinates and size of each linestring
def get_segments_idx(coords, tal_lengths):
    to_remove = []
    acc = 0
    for e in tal_lengths[:-1]:
        to_remove.append((e + acc -1, e + acc))
        acc += e
    segs = [(i, i + 1) for i in range(len(coords)-1) if (i, i + 1) not in to_remove]
    return segs

# length of edge referenced by index idx (pt_ini, pt_fin) in pts
def edge_length(idx, pts):
    xa, ya, xb, yb = pts[idx[0]*2], pts[idx[0]*2 + 1], pts[idx[1]*2], pts[idx[1]*2 + 1]
    return ((xa - xb)**2 + (ya- yb)**2)**0.5

# num of the linestring for point at index idx 
def num_talus(idx, talus_lengths):
    i = 0
    s = talus_lengths[0]
    while idx >= s:
        i += 1
        s += talus_lengths[i]
    return i

# remove edges where angles are too narrow
def decimate_edges(edges, vertices, talus_lengths, EPSILON = 0.11):
    remainings = set()
    for e in edges:
        te1 = num_talus(e[0], talus_lengths) 
        te2 = num_talus(e[1], talus_lengths)
        if abs(e[0] - e[1]) ==  1 or te1 != te2:
            remainings.add(tuple(e))
    for i in range(sum(talus_lengths)):
        sub_i = [e for e in remainings if e[0] == i or e[1] == i]
        #print(sub_i)
        for j in range(len(sub_i) - 1):
            ej = sub_i[j]
            for k in range(j + 1, len(sub_i)):
                ek = sub_i[k]
                points_idx = {ej[0], ej[1], ek[0], ek[1]} # 3, one point is common to two edges
                nb_talus = len( {num_talus(point_idx, talus_lengths) for point_idx in points_idx} )
                if nb_talus == 1 : # same talus, edges are consecutives, we keep them
                    continue
                u = vertices[ej[0]] - vertices[ej[1]]
                v = vertices[ek[0]] - vertices[ek[1]]
                ratio = np.abs(np.cross(u,v) / (np.linalg.norm(u) * np.linalg.norm(v)))
                #print(u, '^', v, 'ratio ==', '§§', ratio, '::', nb_talus, len(points_idx))
                if ratio < EPSILON :
                    ej_on_two_talus = num_talus(ej[0], talus_lengths) != num_talus(ej[1], talus_lengths)
                    ek_on_two_talus = num_talus(ek[0], talus_lengths) != num_talus(ek[1], talus_lengths)
                    if ej_on_two_talus and ek_on_two_talus:
                        if ej in remainings:
                            remainings.remove(ej)
                        elif ek in remainings:
                            remainings.remove(ek)
    return np.array(list(remainings))

def get_edges_from_triangulation(points_talus, talus_lengths, decimate=True):
    """ returns a list of edges (pt_ini, pt_fin) from the triangulation of points_talus
    """
    segs = get_segments_idx(points_talus, talus_lengths)
    dataset = dict()
    dataset["vertices"] = points_talus
    dataset["segments"] = segs
    t = tr.triangulate(dataset, 'e')
    edges = t['edges']
    if decimate:
        edges = decimate_edges(edges, points_talus, talus_lengths)
    return edges
