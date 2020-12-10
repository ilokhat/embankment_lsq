import numpy as np
import triangle as tr


# returns a list of internal segments (pt_ini, pt_fin) from a list of coordinates and size of each linestring
# removing the segments between each linestrings
def get_segments_idx(coords, tal_lengths):
    to_remove = []
    acc = 0
    for e in tal_lengths[:-1]:
        to_remove.append((e + acc - 1, e + acc))
        acc += e
    segs = [(i, i + 1) for i in range(len(coords)-1) if (i, i + 1) not in to_remove]
    return segs

# length of edge referenced by index idx (pt_ini, pt_fin) in pts
def edge_length(idx, pts):
    xa, ya, xb, yb = pts[idx[0]*2], pts[idx[0]*2 + 1], pts[idx[1]*2], pts[idx[1]*2 + 1]
    return ((xa - xb)**2 + (ya - yb)**2)**0.5

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
        if abs(e[0] - e[1]) ==  1 or te1 != te2:  # consecutive points or from 2 different lines
            remainings.add(tuple(e))
    # for i in range(sum(talus_lengths)):
    #     sub_i = [e for e in remainings if e[0] == i or e[1] == i]
    #     #print(sub_i)
    #     for j in range(len(sub_i) - 1):
    #         ej = sub_i[j]
    #         for k in range(j + 1, len(sub_i)):
    #             ek = sub_i[k]
    #             points_idx = {ej[0], ej[1], ek[0], ek[1]} # 3, one point is common to two edges
    #             nb_talus = len( {num_talus(point_idx, talus_lengths) for point_idx in points_idx} )
    #             if nb_talus == 1 : # same talus, edges are consecutives, we keep them
    #                 continue
    #             u = vertices[ej[0]] - vertices[ej[1]]
    #             v = vertices[ek[0]] - vertices[ek[1]]
    #             ratio = np.abs(np.cross(u,v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    #             print(u, '^', v, 'ratio ==', '§§', ratio, '::', nb_talus, len(points_idx))
    #             if ratio < EPSILON :
    #                 ej_on_two_talus = num_talus(ej[0], talus_lengths) != num_talus(ej[1], talus_lengths)
    #                 ek_on_two_talus = num_talus(ek[0], talus_lengths) != num_talus(ek[1], talus_lengths)
    #                 if ej_on_two_talus and ek_on_two_talus:
    #                     if ej in remainings:
    #                         remainings.remove(ej)
    #                     elif ek in remainings:
    #                         remainings.remove(ek)
    return np.array(list(remainings))

def get_edges_from_triangulation(points_talus, talus_lengths, decimate=False):
    """ returns a list of edges (pt_ini, pt_fin) from the triangulation of points_talus
    """
    vertices = points_talus.reshape(-1, 2)
    segs = get_segments_idx(vertices, talus_lengths)
    dataset = dict()
    dataset["vertices"] = vertices
    dataset["segments"] = segs
    t = tr.triangulate(dataset, 'e')
    edges = t['edges']
    if decimate:
        edges = decimate_edges(edges, vertices, talus_lengths)
    return edges


if __name__ == "__main__":
    from shapely.wkt import loads
    from shapes_and_geoms_stuff import get_points_talus

    DECIMATE_EDGES = False
    talus = [#'LineString(920235.39999999850988388 6311461.89999999850988388 578.10000000000582077, 920243.79999999701976776 6311459.5 574.30000000000291038, 920252.89999999850988388 6311457.69999999925494194 571.69999999999708962, 920270.60000000149011612 6311455.39999999850988388 567.39999999999417923, 920296.5 6311455.5 559.89999999999417923, 920304.10000000149011612 6311454.39999999850988388 557.30000000000291038, 920310.89999999850988388 6311452 554.60000000000582077, 920329.10000000149011612 6311443.69999999925494194 544.10000000000582077)',
             'LineString(920416.60000000149011612 6311357.60000000149011612 538.39999999999417923, 920408.70000000298023224 6311359.5 542.19999999999708962, 920397.20000000298023224 6311363 547.39999999999417923, 920388.70000000298023224 6311363.60000000149011612 547.30000000000291038, 920373.79999999701976776 6311367.69999999925494194 552, 920368.70000000298023224 6311365.89999999850988388 554.19999999999708962, 920362 6311367.69999999925494194 554.19999999999708962, 920355.5 6311371.5 554.19999999999708962, 920351.70000000298023224 6311376.89999999850988388 554.19999999999708962, 920350.20000000298023224 6311382.19999999925494194 554.19999999999708962, 920344.79999999701976776 6311389.80000000074505806 556.5, 920340.60000000149011612 6311396.39999999850988388 557, 920332 6311399.89999999850988388 559.5, 920322.70000000298023224 6311399.80000000074505806 562.19999999999708962, 920314.29999999701976776 6311400.60000000149011612 565.39999999999417923, 920304.79999999701976776 6311401.10000000149011612 565.39999999999417923, 920294.20000000298023224 6311401.39999999850988388 569, 920285.39999999850988388 6311402.89999999850988388 572.19999999999708962, 920277 6311403.89999999850988388 574.10000000000582077, 920271.5 6311405.89999999850988388 576.10000000000582077, 920267.5 6311411.80000000074505806 576.19999999999708962, 920264.39999999850988388 6311411 576.19999999999708962, 920259.70000000298023224 6311409.30000000074505806 576.19999999999708962, 920256.20000000298023224 6311405.19999999925494194 580.10000000000582077, 920250.79999999701976776 6311403.80000000074505806 580.10000000000582077, 920244.79999999701976776 6311405.69999999925494194 582.69999999999708962, 920239 6311411.10000000149011612 583.5, 920232.60000000149011612 6311417.10000000149011612 583.5)',
             'LineString(920463.89999999850988388 6311417 522.30000000000291038, 920466.60000000149011612 6311400.69999999925494194 518.89999999999417923, 920468.89999999850988388 6311392 517.19999999999708962, 920479.89999999850988388 6311376.10000000149011612 514.60000000000582077, 920486.89999999850988388 6311369.10000000149011612 513, 920493.70000000298023224 6311360.89999999850988388 511.5, 920498 6311353.60000000149011612 510.19999999999708962, 920501.20000000298023224 6311346.30000000074505806 509, 920502.39999999850988388 6311341.19999999925494194 508.69999999999708962)'
             ]
    talus_shapes = [loads(t) for t in talus]
    talus_lengths = [len(t.coords) for t in talus_shapes]
    points_talus = get_points_talus(talus_shapes)
    edges = get_edges_from_triangulation(points_talus, talus_lengths, decimate=DECIMATE_EDGES)             
    for e in edges:
        seg = f'LINESTRING({points_talus[e[0]*2]} {points_talus[e[0]*2 + 1]}, {points_talus[e[1]*2]} {points_talus[e[1]*2 + 1]})'
        print(seg)
    print(points_talus.shape, edges.shape)