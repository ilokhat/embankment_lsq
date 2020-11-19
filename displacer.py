from timeit import default_timer as timer
import logging
import numpy as np

from shapely.geometry import asLineString
from shapely.ops import unary_union

import pygeos

from triangulation import edge_length, num_talus


#log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_format = '%(message)s'
formatter = logging.Formatter(log_format)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

loglsd = logging.getLogger(__name__)
loglsd.addHandler(stream_handler)
loglsd.setLevel(logging.INFO)

class LSDisplacer:
    MAX_ITER = 250
    NORM_DX = 0.3 #0.3 #0.05
    DIST = 'MIN'
    KKT = False
    # constraints
    ID_CONST = True
    ANGLES_CONST = True
    EDGES_CONST = True
    DIST_CONST = True
    # weights and differentiation step
    H = 2.0 #1.0
    PFix = 1.
    PEdges_int = 10 #10
    PEdges_int_non_seg = 5
    PEdges_ext = 2
    Pedges_ext_far = 0
    PAngles = 50 #50 #25
    PDistRoads = 1000

    def __init__(self, points_talus, roads_shapes, talus_lengths, edges, buffer=15, edges_dist_min=10, edges_dist_max=30):
        self.points_talus = points_talus
        self.roads_shapes = [pygeos.io.from_wkt(r) for r in roads_shapes] #roads_shapes
        self.talus_lengths = talus_lengths
        self.edges = edges
        self.buffer = buffer
        self.edges_dist_min = edges_dist_min
        self.edges_dist_max = edges_dist_max
        self.x_courant = self.points_talus.reshape(-1).copy()
        # 2 * nb points talus
        self.nb_vars = len(self.x_courant)
        self.e_lengths_ori = self._get_edges_lengths(self.edges) # np.array(self.e_lengths_ori)
        self.angles_ori = self.angles_crossprod()
        self.P = self.get_P()
        self.meta = {"nb_iters": -1, "time_s": -1, "dx_reached": -1}
        # test to optimize things
        #self.roads_tals = self._idx_talux_per_road()

    def set_params(MAX_ITER=250, NORM_DX=0.3, H=2.0, DIST='MIN', KKT=False,
                   ID_CONST=True, ANGLES_CONST=True, EDGES_CONST=True, DIST_CONST=True,
                   PFix=1., PEdges_int=10, PEdges_ext=2, PAngles=50, Pedges_ext_far=0, PEdges_int_non_seg=5, PDistRoads=1000):
        LSDisplacer.MAX_ITER = MAX_ITER
        LSDisplacer.NORM_DX = NORM_DX
        LSDisplacer.DIST = DIST
        LSDisplacer.KKT = KKT
        LSDisplacer.ID_CONST = ID_CONST
        LSDisplacer.ANGLES_CONST = ANGLES_CONST
        LSDisplacer.EDGES_CONST = EDGES_CONST
        LSDisplacer.DIST_CONST = DIST_CONST
        LSDisplacer.H = H
        LSDisplacer.PFix = PFix
        LSDisplacer.PEdges_int = PEdges_int
        LSDisplacer.PEdges_int_non_seg = PEdges_int_non_seg
        LSDisplacer.PEdges_ext = PEdges_ext
        LSDisplacer.Pedges_ext_far = Pedges_ext_far
        LSDisplacer.PAngles = PAngles
        LSDisplacer.PDistRoads = PDistRoads


    # utility method to build a shapely line for points starting at offset and having size number of points
    def _line_from_points(points, offset, size):
        tal = asLineString(points[offset:offset+size])
        return tal
    

    # pygeos specific, return a multiline from all points of all talus
    def _multiline_from_points(points, talus_lengths):
        offset = 0
        lines = []
        coords = points.reshape(-1, 2)
        for size in talus_lengths:
            l = pygeos.creation.linestrings(coords[offset: offset+size])
            lines.append(l)
            offset += size
        return pygeos.creation.multilinestrings(lines)
    
    # pygeos, returns an array of linestrings from the points of all talus lines
    def _lines_from_points(points, talus_lengths):
        offset = 0
        lines = []
        coords = points.reshape(-1, 2)
        for size in talus_lengths:
            l = pygeos.creation.linestrings(coords[offset: offset+size])
            lines.append(l)
            offset += size
        return np.array(lines)

    def print_linestrings_wkts(self):
        points = self.x_courant.reshape(-1, 2)
        offset = 0
        for size in self.talus_lengths:
            t = LSDisplacer._line_from_points(points, offset, size)
            print(t)
            offset += size
    
    def get_linestrings_wkts(self):
        points = self.x_courant.reshape(-1, 2)
        offset = 0
        lines = []
        for size in self.talus_lengths:
            t = LSDisplacer._line_from_points(points, offset, size)
            lines.append(t.wkt)
            offset += size
        return lines

    # trying to optimize things, does not work
    # for each road, get the index of talus within distance of buffer
    # def _idx_talux_per_road(self):
    #     points = self.x_courant.reshape(-1, 2)
    #     offset = 0
    #     idx = []
    #     for r in self.roads_shapes:
    #         r_t = []
    #         for i, size in enumerate(self.talus_lengths):
    #             t = LSDisplacer._line_from_points(points, offset, size)
    #             if t.distance(r) < self.buffer:
    #                 r_t.append(i)
    #             offset += size
    #         idx.append(r_t)
    #     return idx



    # set edges original lengths, and set minimal distance for too small inter talus edges
    def _get_edges_lengths(self, edges):
        edges_lengths_ori = []
        for e in edges:
            el = edge_length(e, self.points_talus.reshape(-1))
            if el < self.edges_dist_min and num_talus(e[0], self.talus_lengths) != num_talus(e[1], self.talus_lengths):
                loglsd.debug("min reached inter edges")
                el = self.edges_dist_min
            edges_lengths_ori.append(el)
        return np.array(edges_lengths_ori)

    def angles_crossprod(self):
        """ returns normalized crossproduct of all angles for "inside" points in all talus lines
        """
        offset = 0
        cross_products = []
        for size in self.talus_lengths:
            for i in range(offset + 1, (offset + size - 1)):
                prec = np.array((self.x_courant[2*i - 2], self.x_courant[2*i - 1]))
                pt = np.array((self.x_courant[2*i], self.x_courant[2*i + 1]))
                suiv = np.array((self.x_courant[2*i + 2], self.x_courant[2*i + 3]))
                u = (pt - prec) #/ np.linalg.norm(pt - prec)
                u = u / np.linalg.norm(u)
                v = (suiv - pt) #/ np.linalg.norm(suiv - pt)
                v = v / np.linalg.norm(v)
                cross_products.append(np.cross(u, v))
            offset += size
        return np.array(cross_products)


    # def partial_derivatives_cross_normo(self): #, points, talus_lengths):
    #     offset = 0
    #     cross_products = []
    #     for size in self.talus_lengths:
    #         m = np.zeros(self.nb_vars)
    #         for i in range(offset + 1, offset + size - 1):
    #             u = self.x_courant[2*i - 2:2*i + 4]
    #             # df en Xi-1, Yi-1
    #             dfx = -0.5*(((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*((-u[0] + u[2])*(-u[3] + u[5]) - (-u[1] + u[3])*(-u[2] + u[4]))*(2*u[0] - 2*u[2])/((-u[0] + u[2])**2 + (-u[1] + u[3])**2) + (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(u[3] - u[5])
    #             dfy = -0.5*(((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*((-u[0] + u[2])*(-u[3] + u[5]) - (-u[1] + u[3])*(-u[2] + u[4]))*(2*u[1] - 2*u[3])/((-u[0] + u[2])**2 + (-u[1] + u[3])**2) + (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(-u[2] + u[4])
    #             m[2*i - 2] = dfx
    #             m[2*i - 1] = dfy
    #             # df en Xi, Yi
    #             dfx = (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(-0.5*((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*(2*u[2] - 2*u[4]) - 0.5*((-u[2] + u[4])**2 + (-u[3] + u[5])**2)*(-2*u[0] + 2*u[2]))*((-u[0] + u[2])*(-u[3] + u[5]) - (-u[1] + u[3])*(-u[2] + u[4]))/(((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2)) + (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(-u[1] + u[5])
    #             dfy = (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(-0.5*((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*(2*u[3] - 2*u[5]) - 0.5*((-u[2] + u[4])**2 + (-u[3] + u[5])**2)*(-2*u[1] + 2*u[3]))*((-u[0] + u[2])*(-u[3] + u[5]) - (-u[1] + u[3])*(-u[2] + u[4]))/(((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2)) + (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(u[0] - u[4])
    #             m[2*i] = dfx
    #             m[2*i + 1] = dfy
    #             # df en Xi+1, Yi+1
    #             dfx = -0.5*(((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*((-u[0] + u[2])*(-u[3] + u[5]) - (-u[1] + u[3])*(-u[2] + u[4]))*(-2*u[2] + 2*u[4])/((-u[2] + u[4])**2 + (-u[3] + u[5])**2) + (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(u[1] - u[3])
    #             dfy = -0.5*(((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*((-u[0] + u[2])*(-u[3] + u[5]) - (-u[1] + u[3])*(-u[2] + u[4]))*(-2*u[3] + 2*u[5])/((-u[2] + u[4])**2 + (-u[3] + u[5])**2) + (((-u[0] + u[2])**2 + (-u[1] + u[3])**2)*((-u[2] + u[4])**2 + (-u[3] + u[5])**2))**(-0.5)*(-u[0] + u[2])
    #             m[2*i + 2] = dfx
    #             m[2*i + 3] = dfy
    #             cross_products.append(m)
    #         offset += size
    #     return np.array(cross_products)

    # derived with sympy
    def partial_derivatives_cross_norm(self):
        offset = 0
        cross_products = []
        for size in self.talus_lengths:
            m = np.zeros(self.nb_vars)
            for i in range(offset + 1, offset + size - 1):
                # xi, yi => x, y | xi-1, yi-1 => xpp, ypp | xi+1, yi+1 => xss, yss
                u = self.x_courant[2*i - 2:2*i + 4]
                xpp, ypp, x, y, xss, yss = u[0], u[1], u[2], u[3], u[4], u[5]
                # df/dxi-1
                m[2*i - 2] = (((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))**(-0.5)*(-1.0*(x - xpp)*((x - xpp)*(y - yss) - (x - xss)*(y - ypp)) + (y - yss)*((x - xpp)**2 + (y - ypp)**2))/((x - xpp)**2 + (y - ypp)**2)
                # df/dyi-1
                m[2*i - 1] = (((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))**(-0.5)*((-x + xss)*((x - xpp)**2 + (y - ypp)**2) - 1.0*(y - ypp)*((x - xpp)*(y - yss) - (x - xss)*(y - ypp)))/((x - xpp)**2 + (y - ypp)**2)
                # df/dxi
                m[2*i] = (((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))**(-0.5)*((-ypp + yss)*((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2) + ((x - xpp)*(y - yss) - (x - xss)*(y - ypp))*((x - xpp)*((x - xss)**2 + (y - yss)**2) + (x - xss)*((x - xpp)**2 + (y - ypp)**2)))/(((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))
                # df/dyi
                m[2*i + 1] = (((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))**(-0.5)*((xpp - xss)*((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2) + ((x - xpp)*(y - yss) - (x - xss)*(y - ypp))*((y - ypp)*((x - xss)**2 + (y - yss)**2) + (y - yss)*((x - xpp)**2 + (y - ypp)**2)))/(((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))
                # df/dxi+1
                m[2*i + 2] = (((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))**(-0.5)*(-1.0*(x - xss)*((x - xpp)*(y - yss) - (x - xss)*(y - ypp)) + (-y + ypp)*((x - xss)**2 + (y - yss)**2))/((x - xss)**2 + (y - yss)**2)
                # df/dyi+1
                m[2*i + 3] = (((x - xpp)**2 + (y - ypp)**2)*((x - xss)**2 + (y - yss)**2))**(-0.5)*((x - xpp)*((x - xss)**2 + (y - yss)**2) - 1.0*(y - yss)*((x - xpp)*(y - yss) - (x - xss)*(y - ypp)))/((x - xss)**2 + (y - yss)**2)                
                cross_products.append(m)
            offset += size
        return np.array(cross_products)
    
    # original
    # def dist_F_original(self, road, points): # tal_lengths
    #     min_dist = np.inf if LSDisplacer.DIST == 'MIN' else 0
    #     offset = 0
    #     for i, size in enumerate(self.talus_lengths):
    #         t = LSDisplacer._line_from_points(points, offset, size)
    #         if LSDisplacer.DIST == 'MIN':
    #             min_dist = min(t.distance(road), min_dist)  
    #         else:
    #             min_dist += t.distance(road)
    #         offset += size
    #     if LSDisplacer.DIST != 'MIN':
    #         min_dist /= len(self.talus_lengths)
    #     dist = 0. if min_dist >= self.buffer else (self.buffer - min_dist) #**2
    #     return dist
    
    # test small optim
    def dist_F(self, road, points): #tal_lengths
        min_dist = 0
        offset = 0
        if LSDisplacer.DIST == 'MIN':
            tals = []
            for i, size in enumerate(self.talus_lengths):
                tals.append(LSDisplacer._line_from_points(points, offset, size))
                offset += size
            tals = unary_union(tals)
            min_dist = road.distance(unary_union(tals))
        else:
            for i, size in enumerate(self.talus_lengths):
                t = LSDisplacer._line_from_points(points, offset, size)
                min_dist += t.distance(road)
                offset += size
            min_dist /= len(self.talus_lengths)
        dist = 0. if min_dist >= self.buffer else (self.buffer - min_dist) #**2
        return dist
    
    # each line of points_array contains points for multiple lines, offset and size is deduced from self.talus_lengths
    # returns an array of distances from road, either min or max
    def dist_F_vectorized(self, road, points_array):
        ml = []
        for c in points_array:
            if LSDisplacer.DIST == 'MIN':
                m = LSDisplacer._multiline_from_points(c, self.talus_lengths)
            else:
                m = LSDisplacer._lines_from_points(c, self.talus_lengths) 
            ml.append(m)
        ml = np.array(ml)
        dists = pygeos.distance(road, ml)
        if LSDisplacer.DIST != 'MIN':
            dists = dists.mean(axis=1)
        dists = np.where(dists > self.buffer, 0., self.buffer - dists)
        return dists

    # original
    def dist_F_derivative__(self, road):
        coords = self.x_courant.reshape(-1, 2)
        l = np.zeros(self.nb_vars)
        for i in range(self.nb_vars):
            hi = np.zeros(self.nb_vars)
            hi[i] = self.H
            hi = hi.reshape(-1, 2)
            dist = (self.dist_F(road, coords + hi) - self.dist_F(road, coords - hi)) / (2. * self.H)
            l[i] = dist
        return l

    # def dist_F_derivative_(self, road):
    #     coords = self.x_courant
    #     # diagonal matrix with H on diagonal
    #     h = np.eye(self.nb_vars) * self.H
    #     coords_plus_H = self.x_courant + h
    #     coords_minus_H = self.x_courant - h
    #     d_plus = np.zeros(self.nb_vars)
    #     d_min = np.zeros(self.nb_vars)
    #     for i in range(self.nb_vars):
    #         d_plus[i] = self.dist_F(road, coords_plus_H[i].reshape(-1, 2))
    #         d_min[i] = self.dist_F(road, coords_minus_H[i].reshape(-1, 2))
    #     return (d_plus - d_min) / (2* self.H)

    def dist_F_derivative(self, road):
        #pyr = pygeos.io.from_wkt(road.wkt)
        coords = self.x_courant
        # diagonal matrix with H on diagonal
        h = np.eye(self.nb_vars) * self.H
        coords_plus_H = self.x_courant + h
        coords_minus_H = self.x_courant - h
        # conc = np.concatenate((coords_plus_H, coords_minus_H))
        # ml = []
        # for c in conc:
        #     m = LSDisplacer._multiline_from_points(c, self.talus_lengths)
        #     ml.append(m)
        # ml = np.array(ml)
        # ds = pygeos.distance(pyr, ml)
        # ds = np.where(ds > self.buffer, 0., self.buffer - ds)
        # ds = (ds[:self.nb_vars] - ds[self.nb_vars:]) / (2* self.H)

        # seems a bit faster to have 2 np arrays instead of the same one splitted       
        d_plus = self.dist_F_vectorized(road, coords_plus_H)
        d_min = self.dist_F_vectorized(road, coords_minus_H)
        ds = (d_plus - d_min) / (2* self.H)
        return ds

    # no good
    # def dist_F_derivative_optim(self, road, ir):
    #     coords = self.x_courant.reshape(-1, 2)
    #     l = np.zeros(self.nb_vars)
    #     for i in range(self.nb_varsœ):
    #         if num_talus(i//2, self.talus_lengths) not in self.roads_tals[ir]:
    #             continue
    #         hi = np.zeros(self.nb_vars)
    #         hi[i] = self.H
    #         hi = hi.reshape(-1, 2)
    #         dist = (self.dist_F(road, coords + hi) - self.dist_F(road, coords - hi)) / (2. * self.H)
    #         l[i] = dist
    #     return l

    # idx : edge index in pts [idx_p1, idx_p2]
    def edge_length_diff(self, idx):
        xa, ya, xb, yb = self.x_courant[idx[0]*2], self.x_courant[idx[0]*2 + 1], self.x_courant[idx[1]*2], self.x_courant[idx[1]*2 + 1]
        ab = ((xa - xb)**2 + (ya - yb)**2) ** 0.5
        l = np.zeros(self.nb_vars)
        l[idx[0] * 2] = (xa - xb) / ab
        l[idx[0] * 2 + 1] = (ya - yb) / ab
        l[idx[1] * 2] = -l[idx[0] * 2] # -(xa - xb) / ab
        l[idx[1] * 2 + 1] = -l[idx[0] * 2 + 1] #- (ya - yb) / ab
        return l

    def get_P(self):
        weights = []
        if LSDisplacer.ID_CONST:
            wfix = np.full(2*len(self.points_talus), LSDisplacer.PFix)
            weights.append(wfix)
        if LSDisplacer.ANGLES_CONST:
            wAngles = np.full(len(self.angles_ori), LSDisplacer.PAngles)
            weights.append(wAngles)
        if LSDisplacer.EDGES_CONST:
            wEdges = []
            for i, e in enumerate(self.edges):
                same_talus = num_talus(e[0], self.talus_lengths) == num_talus(e[1], self.talus_lengths)
                non_consecutive_points = abs(e[0] - e[1]) != 1
                if same_talus:
                    if non_consecutive_points:
                        loglsd.debug("**** non intra edge segment : limiting weight")
                        wEdges.append(LSDisplacer.PEdges_int_non_seg)
                    else:
                        loglsd.debug("**** intra edge segment")
                        wEdges.append(LSDisplacer.PEdges_int)
                else:
                    if edge_length(e, self.points_talus.reshape(-1)) >= self.edges_dist_max:
                        loglsd.debug("**** max inter edges threshold reached : minimalizing weight for this edge")
                        wEdges.append(LSDisplacer.Pedges_ext_far)
                    else:
                        loglsd.debug("**** intra edges segment")
                        wEdges.append(LSDisplacer.PEdges_ext)
            wEdges = np.array(wEdges)
            weights.append(wEdges)
        if LSDisplacer.DIST_CONST and not LSDisplacer.KKT:
            wRoads = np.full(len(self.roads_shapes), LSDisplacer.PDistRoads)
            weights.append(wRoads)
        return np.diag(np.concatenate(weights))


    # B = Y - S(Xcourant)
    def get_B(self): 
        b = None
        # inertia
        if LSDisplacer.ID_CONST:
            b = self.points_talus.reshape(-1) - self.x_courant
        # cross prod angles
        if LSDisplacer.ANGLES_CONST:
            angles_cross = self.angles_crossprod()
            if len(angles_cross) != 0:
                if b is None:
                    b = self.angles_ori - angles_cross
                else:
                    b = np.concatenate((b, (self.angles_ori - angles_cross)))
        # triangulation edges
        if LSDisplacer.EDGES_CONST:
            e_lengths = []
            for e in self.edges:
                e_lengths.append(edge_length(e, self.x_courant))
            ee = self.e_lengths_ori - np.array(e_lengths)
            if b is None:
                b = ee
            else:
                b = np.concatenate((b, ee))
        # distance from roads
        if LSDisplacer.DIST_CONST:
            r_dists = []
            for r in self.roads_shapes:
                #fk = - self.dist_F(r, self.x_courant.reshape(-1, 2))
                fk = - self.dist_F_vectorized(r, self.x_courant[np.newaxis,:] )
                r_dists.append(fk.item())
            if b is None:
                b = np.array(r_dists)
            else:
                b = np.concatenate((b, r_dists))
        return b


    def get_A(self): #, X_courant, roads_shapes, talus_lengths, edges): #, ID_CONST=True, ANGLES_CONST=True, EDGES_CONST=True, DIST_CONST=True):
        a = None
        # inertia
        if LSDisplacer.ID_CONST:
            a = np.identity(self.nb_vars)
        # cross prod angles
        if LSDisplacer.ANGLES_CONST:
            angles = self.partial_derivatives_cross_norm()
            if len(angles) != 0:
                if a is None:
                    a = angles
                else:
                    a = np.vstack((a, angles))
        # triangulation edges
        if LSDisplacer.EDGES_CONST:
            e_lens=[]
            for e in self.edges:
                ek = self.edge_length_diff(e)
                e_lens.append(ek)
            if a is None:
                a = np.array(e_lens)
            else:
                a = np.vstack((a, e_lens))
        # distance from roads
        if LSDisplacer.DIST_CONST:
            r_dists = []
            # test optimiation
            #for i, r in enumerate(self.roads_shapes):
            for r in self.roads_shapes:
                #fk = self.dist_F_derivative_optim(r, i)
                fk = self.dist_F_derivative(r)
                r_dists.append(fk)
            if a is None :
                a = np.array(r_dists)
            elif len(r_dists) != 0:
                a = np.vstack((a, r_dists))
        return a

    def compute_dx(self):
        if LSDisplacer.KKT:
            idsave, angsave, edgessave = LSDisplacer.ID_CONST, LSDisplacer.ANGLES_CONST, LSDisplacer.EDGES_CONST
            LSDisplacer.DIST_CONST = False
            A = self.get_A()
            B = self.get_B()
            LSDisplacer.DIST_CONST = True
            LSDisplacer.ID_CONST, LSDisplacer.ANGLES_CONST, LSDisplacer.EDGES_CONST = False, False, False
            C = self.get_A()
            D = self.get_B()
            LSDisplacer.ID_CONST, LSDisplacer.ANGLES_CONST, LSDisplacer.EDGES_CONST = idsave, angsave, edgessave

            loglsd.debug(f"A{A.shape} B{B.shape} C{C.shape} D{D.shape} P{self.P.shape}")
            atp = A.T @ self.P 
            atpa = atp @ A
            kkt = np.vstack((2*atpa, C))
            kkt = np.hstack((kkt, np.vstack((C.T, np.zeros((len(self.roads_shapes), len(self.roads_shapes)))))))
            atpb = atp @ B
            kkt_b = np.concatenate((2* atpb, D))
            dx = np.linalg.lstsq(kkt, kkt_b, rcond=None)
            return dx
        A = self.get_A()
        B = self.get_B()
        loglsd.debug(f"A{A.shape} B{B.shape} P{self.P.shape}")
        atp = A.T @ self.P 
        atpa = atp @ A
        atpb = atp @ B
        dx = np.linalg.lstsq(atpa, atpb, rcond=None)
        return dx

    def square(self):
        """
        returns a numpy array of the displaced points after the least square process
        """
        if LSDisplacer.KKT:
            loglsd.info("mode KKT")
            LSDisplacer.DIST_CONST = False
        alpha = 0.1
        ro = 0.1
        min_dx = np.inf
        norm_float = LSDisplacer.NORM_DX
        start_loop = timer()
        i = 0
        for i in range(LSDisplacer.MAX_ITER):
            start = timer()
            dx = self.compute_dx()
            if LSDisplacer.KKT:
                self.x_courant += alpha * dx[0][:len(self.points_talus)*2]
            else :
                self.x_courant += alpha * dx[0] #dx
            end = timer()
            normdx = np.linalg.norm(dx[0], ord=np.inf)
            alpha = (LSDisplacer.H * ro) / (2**0.5 * normdx) if normdx != 0 else 0.1
            min_dx = normdx if normdx < min_dx else min_dx
            loglsd.info(f'iter {i}/ |dx| : {normdx:.4f} -- NORM_DXf: {norm_float} -- mean(dx): {np.mean(dx[0]):.2f} -- {(end - start):.2f}s per step')
            if normdx < norm_float : #NORM_DX :
                break
            norm_float = LSDisplacer.NORM_DX if i < 100 else (LSDisplacer.NORM_DX + 2 * min_dx) / 3
        end_loop = timer()
        self.meta['nb_iters'], self.meta['time_s'], self.meta['dx_reached'] = i, (end_loop - start_loop), min_dx
        loglsd.warning(f'nb iterations: {i + 1} -- min |dx| reached: {min_dx} -- NORM_DXf: {norm_float} -- {(self.meta["time_s"]):.2f}s ')
        return self.x_courant

if __name__ == '__main__':
    import fiona
    from triangulation import get_edges_from_triangulation
    from shapes_and_geoms_stuff import get_STRtrees, get_roads_for_face, get_talus_inside_face, get_points_talus

    faces_file = "/home/imran/projets/talus/Donnees_talus/Talus/faces_reseau.shp"
    network_file = "/home/imran/projets/talus/Donnees_talus/Talus/reseaux_fusionnes.shp"
    talus_file = "/home/imran/projets/talus/Donnees_talus/o_ligne_n0.shp"

    DECIMATE_EDGES = False
    MAX_MAT_SIZE = 500 #300 #1200
    FACE = 1641 #1153 #752
    LSDisplacer.KKT = False

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
        
        displacer = LSDisplacer(points_talus, roads_shapes, talus_lengths, edges, 15)

        p = displacer.get_P()
        if p.shape[0] > MAX_MAT_SIZE: #300:
            print("Skipped, big matrix", p.shape[0])
            print('----------------------------------------------------------------------')
            continue
        print("P shape: ", p.shape[0])
        res = displacer.square()
        #res = res.reshape(-1, 2)
        displacer.print_linestrings_wkts()
        print('----------------------------------------------------------------------')

    end = timer()
    print(f"done in {(end - start):.0f} s")
    faces.close()

