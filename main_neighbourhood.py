import logging
from timeit import default_timer as timer
import fiona
from shapely.geometry import shape, MultiPolygon
from shapely.wkt import loads
from shapely.strtree import STRtree
from shapely.ops import unary_union
from shapes_and_geoms_stuff import get_STRtrees, get_roads_for_face, get_talus_inside_face, get_points_talus, merge_roads
from triangulation import get_edges_from_triangulation
from displacer import LSDisplacer, loglsd

faces_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/faces_reseau.shp"
network_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/reseaux_fusionnes.shp"
talus_file = "/mnt/data/mac/work/talus/Donnees_talus/o_ligne_n0.shp"

LSDisplacer.set_params(MAX_ITER=250, DIST='MIN', ANGLES_CONST=True, NORM_DX=0.3,
                       PAngles=50, PEdges_ext=50, Pedges_ext_far=1, PEdges_int=5, PEdges_int_non_seg=2, PDistRoads=1000)
loglsd.setLevel(logging.WARNING) # par défaut on est en level INFO
#loglsd.setLevel(logging.DEBUG)

MAX_MAT_SIZE = 400 #400 #550 #400
FACE = 514 #3032 #3938 #3994 #3247 #4262 #3550 #6850 #8942 #8890 #1641 #1153 #752
DECIMATE_EDGES = False
BUF = 15 # 6.5
EDGES_D_MIN = 10.
EDGES_D_MAX = 30.

# on log si on est en warning level plus restrictif, on ne log pas les itérations pour chaque calcul par ex..
if loglsd.level == logging.WARNING:
    logfile = f'out_a_{LSDisplacer.PAngles}_eext_{LSDisplacer.PEdges_ext}_eextfar_{LSDisplacer.Pedges_ext_far}_eint_{LSDisplacer.PEdges_int}_eint_ns_{LSDisplacer.PEdges_int_non_seg}.log'
    fh = logging.FileHandler(logfile)
    loglsd.addHandler(fh)
faces = fiona.open(faces_file, 'r')
ntree, ttree = get_STRtrees(network_file, talus_file)
start = timer()

def get_shapes_partition(buff_union, shapes):
    groups = []
    if buff_union.geom_type == 'Polygon':
        buff_union = MultiPolygon([buff_union])
    # reusing a tree could have been more efficient, 
    # but it's small and it didn't work as expected when I tested
    for p in buff_union :
        g = []
        done = set()
        for i, t in enumerate(shapes):
            if i not in done and t.intersects(p):
                g.append(t)
                done.add(i)
        groups.append(g)
    return groups

for i, f in enumerate(faces):
    # if i != FACE:
    #     continue
    if i <= 3000:
        continue
    # if i > 3000:
    #     break
    roads_shapes = get_roads_for_face(f, ntree, merge=False)
    # for r in roads_shapes:
    #     print(r)
    talus_shapes = get_talus_inside_face(f, ttree, merge=True, displace=True)
    #idt_by_wkt = [dict((t.wkt, i) for i, t in enumerate(talus_shapes))]
    # for t in talus_shapes:
    #     print(t)
    #talus_shapes.pop(0)
    # small_ttree = STRtree(talus_shapes)
    # small_rtree = STRtree(roads_shapes)
    # msg = f"small talus tree (size {len(talus_shapes)}) built"
    # loglsd.warning(msg)
    
    u = unary_union([t.buffer(EDGES_D_MAX) for t in talus_shapes])
    tals_groups = get_shapes_partition(u, talus_shapes)
    roads_groups = get_shapes_partition(u, roads_shapes)
    roads_groups = [merge_roads(rg) for rg in roads_groups]

    #print(len(tals_groups), len(roads_groups))
    # for j in range(len(tals_groups)):
    #     print(len(tals_groups[j]), len(roads_groups[j]))
    #break
    # idx = 8
    # talus_shapes = tals_groups[idx]
    # roads_shapes = roads_groups[idx]

    for idx in range(len(tals_groups)):
        talus_shapes = tals_groups[idx]
        roads_shapes = roads_groups[idx]
        nb_tals = len(talus_shapes)
        talus_lengths = [len(t.coords) for t in talus_shapes]

        msg = f'Face {i} idx {idx} | nb talus: {nb_tals} | nb points talus: {sum(talus_lengths)} | nb roads: {len(roads_shapes)}'
        loglsd.warning(msg)

        # on ne traite pas les faces sans talus, celles ou il y a 2 points(pas de triangulation), ou trop grosses
        if nb_tals == 0 or sum(talus_lengths) == 2 or len(roads_shapes) > 10 or len(talus_shapes) > 10 : 
            loglsd.warning('Skipped, no talus or only 2 points or too much roads or talus')
            loglsd.warning('----------------------------------------------------------------------')
            continue
        
        points_talus = get_points_talus(talus_shapes)
        edges = get_edges_from_triangulation(points_talus, talus_lengths, decimate=DECIMATE_EDGES)

        nb_angles = len(points_talus) - 2 * nb_tals #len(angles_crossprod(points_talus.reshape(-1), talus_lengths))
        msg = f'nb angles: {nb_angles} | nb edges selected: {len(edges)}'
        loglsd.warning(msg)
        
        displacer = LSDisplacer(points_talus, roads_shapes, talus_lengths, edges, buffer=BUF, edges_dist_min=EDGES_D_MIN, edges_dist_max=EDGES_D_MAX)

        p = displacer.get_P()
        msg = f'P shape: {p.shape[0]}'
        loglsd.warning(msg)
        if p.shape[0] > MAX_MAT_SIZE: #300:
            msg = f'Skipped, too big, limit set to {MAX_MAT_SIZE}'
            loglsd.warning(msg)
            loglsd.warning('----------------------------------------------------------------------')
            continue
        
        res = displacer.square()
        for l in displacer.get_linestrings_wkts():
            loglsd.warning(l)
        loglsd.warning('----------------------------------------------------------------------')

end = timer()
loglsd.warning(f"done in {(end - start):.0f} s")
faces.close()


# def get_voisinage(talus_shapes, idx, dist):
#     t = talus_shapes[idx]
#     voisinage = []
#     for i in range(len(talus_shapes)):
#         if i != idx:
#             if t.buffer(dist).intersects(talus_shapes[i]):
#                 voisinage.append(i)
#     return voisinage
