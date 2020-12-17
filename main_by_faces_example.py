import logging
from timeit import default_timer as timer
import fiona
from shapes_and_geoms_stuff import get_STRtrees, get_roads_for_face, get_talus_inside_face, get_points_talus
from triangulation import get_edges_from_triangulation
from displacer import LSDisplacer, loglsd

faces_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/faces_reseau.shp"
network_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/reseaux_fusionnes.shp"
talus_file = "/mnt/data/mac/work/talus/Donnees_talus/o_ligne_n0.shp"

LSDisplacer.set_params(MAX_ITER=250, NORM_DX=0.3,
                       PAngles=50, PEdges_ext=50, Pedges_ext_far=1, PEdges_int=5, PEdges_int_non_seg=2, PDistRoads=1000)
#loglsd.setLevel(logging.WARNING) # par défaut on est en level INFO
loglsd.setLevel(logging.DEBUG)

MAX_MAT_SIZE = 1700 #400
FACE = 5433 #861 #2220 #738 #3032 #423 #3994 #3247 #4262 #3550 #6850 #8942 #8890 #1641 #1153 #752

BUF = 15 # 6.5
EDGES_D_MIN = 10.
EDGES_D_MAX = 30.

# on sauvegarde le fichier de log si on est en level WARNING, moins verbeux que celui par défaut, on ne log pas les itérations pour chaque calcul par ex..
if loglsd.level == logging.WARNING:
    logfile = f'out_a_{LSDisplacer.PAngles}_eext_{LSDisplacer.PEdges_ext}_eextfar_{LSDisplacer.Pedges_ext_far}_eint_{LSDisplacer.PEdges_int}_eint_ns_{LSDisplacer.PEdges_int_non_seg}.log'
    fh = logging.FileHandler(logfile)
    loglsd.addHandler(fh)

faces = fiona.open(faces_file, 'r')
ntree, ttree = get_STRtrees(network_file, talus_file)
start = timer()
for i, f in enumerate(faces):
    if i != FACE:
        continue
    roads_shapes = get_roads_for_face(f, ntree)
    talus_shapes = get_talus_inside_face(f, ttree, merge=True, displace=True)
    nb_tals = len(talus_shapes)
    talus_lengths = [len(t.coords) for t in talus_shapes]
    msg = f'Face {i} | nb talus: {nb_tals} | nb points talus: {sum(talus_lengths)} | nb roads: {len(roads_shapes)}'
    loglsd.warning(msg)
    # on ne traite pas les faces sans talus, celles ou il y a 2 points(pas de triangulation), ou trop grosses
    if nb_tals == 0 or sum(talus_lengths) == 2 or len(roads_shapes) > 20 or len(talus_shapes) > 11 : 
        loglsd.warning('Skipped, no talus or only 2 points or too much roads or talus')
        loglsd.warning('----------------------------------------------------------------------')
        continue
    
    points_talus = get_points_talus(talus_shapes)
    edges = get_edges_from_triangulation(points_talus, talus_lengths)

    nb_angles = len(points_talus) / 2 - 2 * nb_tals #len(angles_crossprod(points_talus.reshape(-1), talus_lengths))
    msg = f'nb angles: {nb_angles} | nb edges selected: {len(edges)}'
    loglsd.warning(msg)
    
    # removed shapely objects from LSDisplacer constructor, wkts and associated buffers expected now
    roads_wkts_and_buffers = [(r.wkt, BUF) for r in roads_shapes]
    displacer = LSDisplacer(points_talus, roads_wkts_and_buffers, talus_lengths, edges, edges_dist_min=EDGES_D_MIN, edges_dist_max=EDGES_D_MAX)

    p = displacer.P
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
