import numpy as np
import fiona
#from shapely.wkt import loads
from shapely.geometry import shape, LineString, Point
from shapely.strtree import STRtree
from shapely.ops import linemerge


def get_points_talus(shapely_lines):
    """ takes a list of shapely linestrings and returns an np array of all coordinates
    from every line
    [[x0, y0], [x1, y1], ... [xn, yn]]
    """
    tals = [np.array(t.coords)[:,:2] for t in shapely_lines] # remove z
    tals = np.concatenate(tals, axis=0)
    return tals

#####
##### face and segments things
#####
def get_talus_inside_face(f, ttree, merge=True, displace=True):
    face = shape(f['geometry'])
    tals_candidates = tals_candidates = ttree.query(face)
    tals = [line for line in tals_candidates if line.intersects(face) and not line.intersection(face).geom_type.endswith('Point')]
    tals = [line if face.contains(line) else line.intersection(face) for line in tals]
    no_multipart = []
    for t in tals:
        if t.geom_type == 'MultiLineString' or t.geom_type == "GeometryCollection" :
            for tt in t:
                if tt.geom_type == 'LineString': # and not face.exterior.contains(tt):
                    no_multipart.append(tt)
        else:
            no_multipart.append(t)
    if merge:
        tals_merged = linemerge(no_multipart)
        if tals_merged.geom_type == 'LineString':
            no_multipart = [tals_merged]
        else: 
            no_multipart = [t for t in tals_merged]
    if displace:
        no_multipart = [displace_line_from_centroid_when_snapped_to_road(face, t) for t in no_multipart]
    return no_multipart

def get_roads_for_face(f, ntree, merge=True):
    face = shape(f['geometry'])
    road_candidates = ntree.query(face)
    roads_shapes = [line for line in road_candidates if line.intersects(face) and not line.intersection(face).geom_type.endswith('Point')]
    if not merge:
        return roads_shapes
    return merge_roads(roads_shapes)
    roads_merged = linemerge(roads_shapes)
    if roads_merged.geom_type == 'LineString':
        roads_shapes = [roads_merged]
    else: 
        roads_shapes = [r for r in roads_merged]
    ext_ints = [LineString(face.exterior)]
    for inte in face.interiors:
        ext_ints.append(LineString(inte))
    rs = ext_ints[:]
    for r in roads_shapes:
        #print(r)
        at_least_one = False
        for l in ext_ints:
            if l.contains(r):
                at_least_one = True
                break
        if not at_least_one:
            rs.append(r)
    return rs

def merge_roads(roads_shapes):
    roads_merged = linemerge(roads_shapes)
    if roads_merged.geom_type == 'LineString':
        roads_shapes = [roads_merged]
    else: 
        roads_shapes = [r for r in roads_merged]
    # ext_ints = [LineString(face.exterior)]
    # for inte in face.interiors:
    #     ext_ints.append(LineString(inte))
    # rs = ext_ints[:]
    # for r in roads_shapes:
    #     #print(r)
    #     at_least_one = False
    #     for l in ext_ints:
    #         if l.contains(r):
    #             at_least_one = True
    #             break
    #     if not at_least_one:
    #         rs.append(r)
    return roads_shapes


def displace_line_from_centroid_when_snapped_to_road(face, line):
    DISPLACEMENT_VECTOR_SIZE = 1. #1
    TOL_SNAPPING = 0.05
    centroid = face.centroid
    #print(centroid)
    new_coords = []
    for c in line.coords:
        if face.exterior.contains(Point(*c)) or face.exterior.distance(Point(*c)) <= TOL_SNAPPING:
            p2d = np.array(c[:2])
            vec = np.array(centroid.coords) - p2d 
            vec = (vec / np.linalg.norm(vec)) * DISPLACEMENT_VECTOR_SIZE
            new_p2d = Point(*(p2d + vec))
            if face.contains(new_p2d):
                #print(f'POINT({new_p2d.x} {new_p2d.y})')
                new_coords.append([new_p2d.x, new_p2d.y, c[2]])
                continue
            else: # on le bouge au hasard pour le faire entrer dans la face
                while True:
                    vec = np.random.randn(2)
                    vec = (vec / np.linalg.norm(vec)) * (DISPLACEMENT_VECTOR_SIZE / 2.)
                    new_p2d = Point(*(p2d + vec))
                    if face.contains(new_p2d):
                        new_coords.append([new_p2d.x, new_p2d.y, c[2]])
                        break
                continue
        new_coords.append(c)
    #print(new_coords)
    return LineString(np.array(new_coords))
#####
##### 
#####

def get_STRtrees(network_file, talus_file):
    network = fiona.open(network_file, 'r')
    talus = fiona.open(talus_file, 'r')
    print("building tree for network of size", len(network))
    nlines = [shape(l['geometry']) for l in network]
    ntree = STRtree(nlines)
    print("tree built --------------------------")
    network.close()
    print(f"building tree for {len(talus)} talus")
    tlines = [shape(t['geometry']) for t in talus]
    ttree = STRtree(tlines)
    print("tree built --------------------------")
    talus.close()
    return ntree, ttree

if __name__ == "__main__":
    faces_file = "/home/imran/projets/talus/Donnees_talus/Talus/faces_reseau.shp"
    network_file = "/home/imran/projets/talus/Donnees_talus/Talus/reseaux_fusionnes.shp"
    talus_file = "/home/imran/projets/talus/Donnees_talus/o_ligne_n0.shp"

    faces = fiona.open(faces_file, 'r')
    FACE = faces[1641]

    ntree, ttree = get_STRtrees(network_file, talus_file)
    print(get_roads_for_face(FACE, ntree))
    print(get_talus_inside_face(FACE, ttree))

