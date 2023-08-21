from typing import Iterable

import numpy as np
import sympy as sp

from matplotlib.patches import Ellipse, Rectangle, PathPatch
from matplotlib.path import Path

class GeoObject:
    def __init__(
        self,
        out_path,
        global_loc,
        inner_hollow=[]):
        
        if isinstance(out_path, Path):
            self.outter_path  = out_path
        else:
            raise ValueError("`out_path` must be matplotlib Path object, and must be closed path.")

        self.inner_hollow = []
        if inner_hollow is not None and len(inner_hollow) !=0:
            self.add_hollows(inner_hollow)
        
        self.global_loc = global_loc
    
    
    def add_hollows(self, hollows:PathPatch):
        for h in hollows:
            if isinstance(h, Path):
                h = PathPatch(h, facecolor="none")
            if isinstance(h, PathPatch):
                self.inner_hollow.append(h)
        return 0
    def del_hollow(self, index=None):
        index_len = len(self.inner_hollow)
        if index is None:
            del(self.inner_hollow[-1])
        elif isinstance(index, int):
            if index_len <= index:
                return 1
            else:
                del(self.inner_hollow[index])
        elif isinstance(Iterable):
            for i in index:
                if index_len >= i and isinstance(i, int):
                    del(self.inner_hollow[i])
        return 0
    
    def add_transform(self, transform, mode="all"):
        if mode=="all":
            self.outter_path = transform.transform_path(self.outter_path)
            for i, hollow in enumerate(self.inner_hollow):
                self.inner_hollow[i] = transform.transform_path(hollow)
        elif mode=="out":
            self.outter_path = transform.transform_path(self.outter_path)
        elif mode =="hollow":
            for i, hollow in enumerate(self.inner_hollow):
                self.inner_hollow[i] = transform.transform_path(hollow)
                
                  
    def get_internal_indices(self, mesh_grid):
        
        nx, ny = mesh_grid[0].shape
        
        points = np.vstack([mesh_grid[0].ravel(), mesh_grid[1].ravel()]).T
        
        p_index_prime = PathPatch(self.outter_path).contains_points(points)
        p_index_inners =[]
        
        for hollow in self.inner_hollow:
            p_index_inners.append(hollow.contains_points(points))
        
        # Union the inner index
        p_index_inner = p_index_inners[0]
        for p in p_index_inners[1:]:
            p_index_inner = np.logical_or(p_index_inner, p)
            
        p_index_inside = np.logical_xor(p_index_prime, p_index_inner)
        
        row_index = (p_index_inside/nx).astype(int)
        col_index = p_index_inside%nx
        
        return row_index, col_index
            
class Source(GeoObject):
    def __init__(self, *args, **kwargs):
        super.__init__(*args, **kwargs)
    
    def source_update(self):
        raise NotImplementedError("")

class Matter(GeoObject):
    def __init__(
        self, 
        local_symbols,
        rel_tensor, 
        *args, **kwargs):
        super.__init__(*args, **kwargs)
        
        self.local_sym = local_symbols
        self.rel_tensor = rel_tensor
        
        self.func_rel_tensor = self.__get_func_gji(local_symbols, rel_tensor)
    
    def __get_func_gji(self, variables, G, modules="numpy"):
        nx, ny = sp.shape(G)
        constant_index = []
        for i in range(0, nx):
            for j in range(0, ny):
                if G[i,j].is_constant():
                    constant_index.append([i, j])
        
        _g_ij = sp.lambdify(# Numpy based calculation module 
            variables, G.tolist(), modules=modules
            ) 
        
        def func_g_ij(*args):
            data_shape = args[0].shape
            dim = len(data_shape)
            ones= np.ones(data_shape)
            result_arr = _g_ij(*args)
            # Fixing ragged nested elements
            for (i, j) in constant_index:
                result_arr[i][j] = result_arr[i][j]*ones # constant * ndarray
            # dtype=objet -> dtype=float
            g_r_arr =[]
            for g in result_arr:
                g_r_arr.append(np.stack(g))
            g_xyz = np.stack(g_r_arr)
            # The returned result is (M.dim, Data.dim) transpose it to (Data.dim, M.dim)
            return np.transpose(g_xyz , (*(list(range(2, dim+2))), 0, 1))
        return func_g_ij
    
    def cal_rel_tensor(self, grid):
        X, Y = grid
        row_i, col_i = self.get_internal_indices(grid)
        gij_xy = self.func_rel_tensor(X[row_i, col_i], Y[row_i, col_i])
        return (row_i, col_i), gij_xy