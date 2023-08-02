from typing import Tuple, Literal, Callable

import sympy as sp
import numpy as np
from matplotlib.patches import Circle, Ellipse, PathPatch
from matplotlib.path import Path

def basic_coord_transform_eqs(vars, cor_i, cor_f):
    p1, p2, p3 = vars
    if cor_i =="xyz":
        if cor_f == "cylinder":
            r0 = sp.sqrt(p1**2 + p2**2)
            eps = [r0, sp.atan(p2/p1), p3]
        elif cor_f == "spherical":
            r0 = sp.sqrt(p1**2 + p2**2)
            eps = [sp.sqrt(r0**2 + p3**2), sp.atan(r0/p3), sp.atan(p2/p1)]
    elif cor_i =="cylinder":
        if cor_f == "xyz":
            eps = [p1*sp.cos(p2), p1*sp.sin(p2), p3]
        elif cor_f == "spherical":
            eps =[sp.sqrt(p1**2 + p3**2), p2, sp.atan(p1/p3)]
    elif cor_i =="spherical":
        if cor_f == "xyz":
            eps = [p1*sp.sin(p2)*sp.cos(p3), 
                   p1*sp.sin(p2)*sp.sin(p3), 
                   p1*sp.cos(p2)]
        elif cor_f == "cylinder":
            r0 = p1 * sp.sin(p3)
            eps = [r0, p2, p1 * sp.cos(p3)]
    return eps

def get_basic_coordinates():
    xyz = sp.symbols("x y z", real = True)
    cylinder = [
        sp.symbols("r", real=True, positive=True),
        sp.symbols("\phi", real=True, positive=True),
        sp.symbols("z", real=True)
    ]
    sph = [
        sp.symbols("r_s", real=True, positive=True),
        sp.symbols("\phi_s", real=True, positive=True),
        sp.symbols(r"\theta_s", real=True, positive=True)
    ]
    return xyz, cylinder, sph
def get_jacobian(qs, eqs): # eps: coordinate representation of p with q. 
    M = []
    for i in range(0, len(eqs)):
        row= []
        for j in range(0, len(qs)):
            row.append(eqs[i].diff(qs[j]))
        M.append(row)
    return sp.Matrix(M)
            
def represent_with_basic_coord(
        jacobian, 
        vars, 
        vars_rep,
        coord_basic:Literal["xyz", "cylinder", "spherical"] = "cylinder",
        coord_rep:Literal["xyz", "cylinder", "spherical"] = "xyz"):
    
    if coord_basic == coord_rep:
        return jacobian
    
    q1, q2, q3 = vars
    eps = basic_coord_transform_eqs(vars_rep, coord_rep, coord_basic)
    return jacobian.subs([(q1, eps[0]), (q2, eps[1]), (q3,eps[2])])


def get_func_gij(
        variables:Tuple[sp.S,...], 
        G:sp.Matrix, 
        modules:Literal["numpy"]='numpy')->Callable:
    """This function returns a fuction which calculates g_ij tensor of 2 dim matrix form.
        Usually, 'lambdify` routine is enough to use for general Numpy data type, however, 
        for matrices returning functions, there is an issue of ragged nested element result.
        This function detecting constant equations in `G` unfurl to general `float` type ndarray.
        with some transformation of index.

    Args:
        variables (Tuple[sp.S,...]): Sympy symbols array. [q1 , q2, ..., qn]
        G (sp.Matrix): Metric tensor definition in 2 dim matrix form. The elements are equations of the variables in first argument symbols.
        modules (Literal[&quot;numpy&quot;], optional): Sympy `lambdify` function argument. Defaults to 'numpy'.

    Returns:
        Callable: A function of calculating g tensor for each X, Y, Z meshgrid points.
    """

    # Detecting constant part. 
    
    nx, ny = sp.shape(G)
    constant_index = []
    for i in range(0, nx):
        for j in range(0, ny):
            if G[i,j].is_constant():
                constant_index.append([i, j])
    
    _g_ij = sp.lambdify(variables, G, modules=modules) # Numpy based calculation module 

    def func_g_ij(*args):
        data_shape = args[0].shape
        dim = len(data_shape)
        ones= np.ones(data_shape)
        result_arr = _g_ij(*args)
        # Fixing ragged nested elements
        for (i, j) in constant_index:
            result_arr[i, j] = result_arr[i, j]*ones # constant * ndarray
        # dtype=objet -> dtype=float
        g_r_arr =[]
        for g in result_arr:
            g_r_arr.append(np.stack(g))
        g_xyz = np.stack(g_r_arr)
        # The returned result is (M.dim, Data.dim) transpose it to (Data.dim, M.dim)
        return np.transpose(g_xyz , (*(list(range(2, dim+2))), 0, 1))

    return func_g_ij


def yee_lattices(X,Y, n=[20, 20]):
    xi,xf = X
    yi,yf = Y

    nx, ny = n

    # Electric field grid
    xline = np.linspace(xi, xf, nx, endpoint=True)
    yline = np.linspace(yi, yf, ny, endpoint=True)

    dx = xline[1]-xline[0]
    dy = yline[1]-yline[0]
    # Magnetic field grid
    xline_m = np.linspace(xi-dx/2, xf+dx/2, nx+1, endpoint=True)
    yline_m = np.linspace(yi-dy/2, yf+dy/2, ny+1, endpoint=True)

    return (np.meshgrid(xline, yline), np.meshgrid(xline_m, yline_m))


#------------------------------------------------------------------
def get_invisible_tensor(
        eq_coor, syms, xyz_vars, basic_coord="cylinder"):
    jaco = get_jacobian(syms, eq_coor)
    coords_eqns = basic_coord_transform_eqs(syms, "cylinder", "xyz")
    coord_t_matrix = get_jacobian(syms, coords_eqns) 

    Ja = jaco@coord_t_matrix
    M = sp.simplify(Ja@Ja.T)/sp.simplify(Ja.det())
    M_xyz = sp.simplify(
    represent_with_basic_coord(
        M, syms, xyz_vars, 
        coord_basic=basic_coord, coord_rep="xyz")
        )
    return M_xyz.inv()

def prepare_simulation(
        space:Tuple[Tuple[float, float], Tuple[float, float]],
        dim:Tuple[int, int],
    ):
    (xi, xf), (yi, yf) = space
    nx, ny = dim
    print(yi,yf)
    E_points, H_points =  yee_lattices((xi, xf), (yi, yf), (nx, ny))

    mesh_shape_e = E_points[0].shape
    mesh_shape_h = H_points[0].shape

    E_field = np.zeros((*mesh_shape_e, 3))
    H_field = np.zeros((*mesh_shape_h, 3))

    I3 = np.eye(3)
    g_tensor_e = np.zeros((*mesh_shape_e, 3, 3))
    g_tensor_h = np.zeros((*mesh_shape_h, 3, 3))

    g_tensor_e[..., :, :] = I3
    g_tensor_h[..., :, :] = I3

    return (E_points, H_points), (E_field, H_field), (g_tensor_e, g_tensor_h)


#---------------------------
"""
r_c, phi_c, z_c = cylin_s

R1, R2 = sp.symbols("R_1 R_2", positive=True, real=True)

u1 = R1 + ((R2- R1)/R1)*r_c
u2 = phi_c
u3 = z_c


cy_xyz = get_jacobian(cylin_s, basic_coord_transform_eqs(cylin_s, "cylinder", "xyz"))
A = get_jacobian([r_c, phi_c, z_c], [u1, u2, u3])
Ja = (A@cy_xyz)
M = sp.simplify(Ja@Ja.T)/sp.simplify(Ja.det())

r1 = 2E-2
r2 = 4E-2

M_cal = M.subs([(R1, r1), (R2, r2)])
M_xyz = sp.simplify(represent_with_basic_coord(M_cal, cylin_s, xyz_s, coord_basic="cylinder", coord_rep="xyz"))
"""