from multiprocessing import Pool, Value, cpu_count
import logging
from timeit import default_timer as timer

import fiona
from shapely.geometry import MultiPolygon
from shapely.ops import unary_union

from shapes_and_geoms_stuff import get_STRtrees, get_roads_for_face, get_talus_inside_face, get_points_talus, merge_roads
from triangulation import get_edges_from_triangulation
from displacer import LSDisplacer, loglsd

faces_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/faces_reseau.shp"
network_file = "/mnt/data/mac/work/talus/Donnees_talus/Talus/reseaux_fusionnes.shp"
talus_file = "/mnt/data/mac/work/talus/Donnees_talus/o_ligne_n0.shp"

LSDisplacer.set_params(MAX_ITER=250, DIST='MIN', NORM_DX=0.3,
                       PFix=8.0, PDistRoads=200,
                       PAngles=8, PEdges_ext=15, Pedges_ext_far=0.5, PEdges_int=1, PEdges_int_non_seg=1)
                       #PAngles=50, PEdges_ext=50, Pedges_ext_far=1, PEdges_int=5, PEdges_int_non_seg=2, PDistRoads=1000)

loglsd.setLevel(logging.WARNING) # par défaut on est en level INFO, et on ne veut pas le détail de chaque itération
#loglsd.setLevel(logging.DEBUG)

MAX_MAT_SIZE = 400 #400
FACE = 9109 #4125 #3247 #6073 #3032 #3938 #3994 #3247 #4262 #3550 #6850 #8942 #8890 #1641 #1153 #752
BUF = 15 + 1.5# 6.5
EDGES_D_MIN = 10.
EDGES_D_MAX = 30.
LSDisplacer.FLOATING_NORM = True
NB_CORES = cpu_count() # 4 chez moi


def get_shapes_partition(buff_union, shapes):
    groups = []
    if buff_union.geom_type == 'Polygon':
        buff_union = MultiPolygon([buff_union])
    # reusing a tree could have been more efficient, 
    # but it's small and it didn't work as expected when I tested
    done = set()
    for p in buff_union :
        g = []
        for i, s in enumerate(shapes):
            if i not in done and s.intersects(p):
                g.append(s)
                done.add(i)
        groups.append(g)
    return groups

def format_res_and_save(res, file):
    with open(file, 'w') as logFile: 
        for r in res:
            for k, v in r.items():
                l = '------------------------------------------------------------------------\n'
                l += f'Face: {k}\n'
                for part in v:
                    l += f'idx: {part["idx"]} -- nb_tals: {part["nb_tals"]} -- nb_roads: {part["nb_roads"]} -- nb_points: {part["nb_pts"]}\n'
                    l += f'nb_angles: {part["nb_angles"]} -- nb_edges: {part["nb_edges"]} -- P size: {part["p_shape"]}\n'
                    l += f'nb_iters: {part["nb_iters"]} -- dx_reached: {part["dx_reached"]} -- time(s): {part["time_s"]}\n'
                    l += part['lines']
                    l += "\n"
                logFile.write(l)
                #print(l)

count_processed = Value('i', 0)

def func(f):
    global count_processed
    fid = f['id']
    roads_shapes = get_roads_for_face(f, ntree, merge=False)
    talus_shapes = get_talus_inside_face(f, ttree, merge=True, displace=True)
    u = unary_union([t.buffer(EDGES_D_MAX) for t in talus_shapes])
    tals_groups = get_shapes_partition(u, talus_shapes)
    roads_groups = get_shapes_partition(u, roads_shapes)
    roads_groups = [merge_roads(rg) for rg in roads_groups]
    res_objs = {fid: []} 
    for idx in range(len(tals_groups)):
        res_obj = {'idx': idx, 'nb_iters': -1, 'time_s': -1, 'dx_reached': -1, 'nb_pts': -1, 'nb_angles': -1, 'nb_edges': -1, 'p_shape': -1, 'lines': ''}
        talus_shapes = tals_groups[idx]
        roads_shapes = roads_groups[idx]
        nb_tals = len(talus_shapes)
        talus_lengths = [len(t.coords) for t in talus_shapes]
        res_obj['nb_tals'] , res_obj['nb_pts'], res_obj['nb_roads'] = nb_tals, sum(talus_lengths), len(roads_shapes)

        # on ne traite pas les faces sans talus, celles ou il y a 2 points(pas de triangulation), ou trop grosses
        if nb_tals == 0 or sum(talus_lengths) == 2 or len(roads_shapes) > 10 or len(talus_shapes) > 10 :
            res_obj['lines'] = 'Skipped, 2 points or too much roads or talus'
            res_objs[fid].append(res_obj)
            continue
        
        points_talus = get_points_talus(talus_shapes)
        edges = get_edges_from_triangulation(points_talus, talus_lengths)
        nb_angles = len(points_talus)//2 - 2 * nb_tals #len(angles_crossprod(points_talus.reshape(-1), talus_lengths))
        res_obj['nb_angles'], res_obj['nb_edges'] = nb_angles, len(edges)
        
        roads_wkts_and_buffers = [(r.wkt, BUF) for r in roads_shapes]
        displacer = LSDisplacer(points_talus, roads_wkts_and_buffers, talus_lengths, edges, edges_dist_min=EDGES_D_MIN, edges_dist_max=EDGES_D_MAX)

        p = displacer.P
        res_obj['p_shape'] = p.shape[0]
        if p.shape[0] > MAX_MAT_SIZE:
            res_obj['lines'] = 'Skipped, too big matrix'
            res_objs[fid].append(res_obj)
            continue
        displacer.square()
        res_obj['nb_iters'], res_obj['time_s'], res_obj['dx_reached'] = displacer.meta['nb_iters'], displacer.meta['time_s'], displacer.meta['dx_reached']
        for l in displacer.get_linestrings_wkts():
            res_obj['lines'] += l + '\n'
        res_objs[fid].append(res_obj)
        with count_processed.get_lock():
            count_processed.value += 1
            print(f'{count_processed.value} done so far, [{fid}] ({res_obj["time_s"]})')
    return res_objs

NB_CORES = cpu_count()
if __name__ == '__main__':
    faces = fiona.open(faces_file, 'r')
    print("Operation runs on", NB_CORES, "cores")
    ntree, ttree = get_STRtrees(network_file, talus_file)

    start = timer()
    with Pool(NB_CORES) as p:
        res = p.map(func, faces)
        #res = p.map(func, faces[FACE:FACE + 1])
    end = timer()
    print(50*"*", len(res))
    params = f'a_{LSDisplacer.PAngles}_eext_{LSDisplacer.PEdges_ext}_eextf_{LSDisplacer.Pedges_ext_far}_eei_{LSDisplacer.PEdges_int}_eeins_{LSDisplacer.PEdges_int_non_seg}_pf_{LSDisplacer.PFix}_F_{FACE}.log'
    format_res_and_save(res, "dt_extended_" + params)
    print(params)
    print(f"done in {(end - start):.0f} s -- {count_processed.value} effectively processed")
    faces.close()

