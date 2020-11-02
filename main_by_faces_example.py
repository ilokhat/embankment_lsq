import logging
from timeit import default_timer as timer
import fiona
from shapes_and_geoms_stuff import get_STRtrees, get_roads_for_face, get_talus_inside_face, get_points_talus
from triangulation import get_edges_from_triangulation
from displacer import LSDisplacer, loglsd

faces_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/faces_reseau.shp"
network_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/reseaux_fusionnes.shp"
talus_file = "/mnt/data/mac/work/talus/Donnees_talus/o_ligne_n0.shp"

LSDisplacer.set_params(MAX_ITER=250, PAngles=50, PEdges_ext=2, PEdges_int=10, PDistRoads=1000, DIST='MIN')
loglsd.setLevel(logging.WARNING)

MAX_MAT_SIZE = 400 #300 #1200
FACE = 7901 #1641 #1153 #752
DECIMATE_EDGES = False
BUF = 15 # 6.5

faces = fiona.open(faces_file, 'r')
ntree, ttree = get_STRtrees(network_file, talus_file)
start = timer()
for i, f in enumerate(faces):
    # if i != FACE:
    #     continue
    # if i < 6308:
    #     continue
    roads_shapes = get_roads_for_face(f, ntree)
    talus_shapes = get_talus_inside_face(f, ttree, merge=True, displace=True)
    nb_tals = len(talus_shapes)
    talus_lengths = [len(t.coords) for t in talus_shapes]
    print("Face", i, "| nb talus:", nb_tals, "| nb points talus:", sum(talus_lengths), "| nb roads:", len(roads_shapes))
    # on ne traite pas les faces sans talus, celles ou il y a 2 points(pas de triangulation), ou trop grosses
    if nb_tals == 0 or sum(talus_lengths) == 2 or len(roads_shapes) > 20 or len(talus_shapes) > 10 : 
        print('Skipped, no talus or only 2 points or too much roads or talus')
        print('----------------------------------------------------------------------')
        continue
    
    points_talus = get_points_talus(talus_shapes)
    edges = get_edges_from_triangulation(points_talus, talus_lengths, decimate=DECIMATE_EDGES)
    nb_angles = len(points_talus) - 2 * nb_tals #len(angles_crossprod(points_talus.reshape(-1), talus_lengths))
    print(f'nb angles: {nb_angles} | nb edges selected: {len(edges)}')
    
    displacer = LSDisplacer(points_talus, roads_shapes, talus_lengths, edges, buffer=BUF)

    p = displacer.get_P()
    if p.shape[0] > MAX_MAT_SIZE: #300:
        print("Skipped, big matrix", p.shape[0])
        print('----------------------------------------------------------------------')
        continue
    print("P shape: ", p.shape[0])
    res = displacer.square()
    displacer.print_linestrings_wkts()
    print('----------------------------------------------------------------------')

end = timer()
print(f"done in {(end - start):.0f} s")
faces.close()
