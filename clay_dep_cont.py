#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 18:26:41 2019

This project implements a continuum model of clay deposition in a sandy bed.

This file provides a library of functions for running the model. The functions are used by run_model.py.

@author: yoni
"""

import numpy as np
from numpy.linalg import norm, det
from scipy.sparse import diags, csc_matrix
from scipy.special import comb
from scipy.interpolate import griddata, interp1d
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from math import ceil
import pandas as pd
from functools import reduce

EXTERIOR_NODE = -1
INTERIOR_NODE = 0
BOUNDARY_NODE = 1

eps = np.finfo(np.float64).eps

# for creating labels when plotting
dh_dz_str = r"$\frac{\partial h}{\partial z}$"

particle_columns = ["X_cm", "Z_cm", "Deposited"]

def flip_arr(arr):
    """
    Flip an array vertically. This makes it more intuitive to view in the variable explorer.
    """
    return arr[::-1,:]

def s_celerity_cm_min(V, h):
    """
    Function to compute bedform celerity from flow velocity and height of water column.
    Uses the formula from Snishchenko & Kopaliani (1978)

    V: flow velocity of surface water, in m/s
    h: height of water column, in m
    """
    g = 9.8 # gravitational constant in m/(s^2)
    Fr = V / np.sqrt(g * h)
    cel_m_s = 0.019 * V * (Fr**2.9)
    cel_cm_min = cel_m_s * 100 * 60
    return cel_cm_min

def get_V_fn(h):
    """
    Returns a function for computing surface water velocity (m/s) based on celerity (cm/min).

    h: height of the water column, in meters
    """
    V_vals_m_s = np.linspace(0, 5, 10000)
    cel_vals_cm_min = s_celerity_cm_min(V_vals_m_s, h)
    V_fn = interp1d(cel_vals_cm_min, V_vals_m_s)
    return V_fn

def get_ElliottBrooks_Hm(flow_velocity_m_s, bedform_height_cm, water_depth_cm):
    """
    # calculate (maximum head for imposed sinusoidal head wave) as (per Elliott 1990).
    """

    g = 9.8 # acceleration due to gravity, in m/s^2
    bedform_height_m = bedform_height_cm / 100
    water_depth_m = water_depth_cm / 100
    ratio = bedform_height_m / water_depth_m
    if ratio <= 0.34:
        exponent = 3/8
    else:
        exponent = 3/2
    H_m = 0.28 * (flow_velocity_m_s**2 / (2*g)) * ((ratio/0.34)**(exponent))
    H_m *= 100 # convert to cm
    return H_m

def exclude_margin(interval, margin, x_axis):
    """
    interval: a 1D numpy array of floats of length 2, specifying the beginning and ending points of an interval along the x-axis of the domain
    margin_cm: a float specifying the magnitude of a margin that we want to exclude from interval near the left or right boundaries
    x_axis: the x-coordinates of nodes along the x-axis of the domain

    Background:
    Head behaves anomalously near the left and right boundaries of the domain. Thus we don't want to consider behavior near these boundaries for the purposes of particle tracking. 'interval' an interval specifying a lee or stoss face of a bedform, as in 'bedform_stoss_ranges', or 'bedform_lee_ranges'. Thus for example if margin_cm is 5 and for 'interval' we have a stoss range of [0,22], this function will curtail that range to [5,22]. If the stoss range is [28, 50], this function will curtail it to [28, 45].
    """
    a = max(interval[0], x_axis.min() + margin)
    b = min(interval[1], x_axis.max() - margin)
    return np.asarray([a,b])

def get_empty_particle_df(particle_columns = particle_columns):
    return pd.DataFrame(columns = particle_columns).astype({"Deposited": bool})

def harm_mean(mats):
    """
    Function to calculate the element-wise harmonic mean of a list of matrices.

    Used for calculating the harmonic mean of K over adjacent nodes, for determining flow paths.
    """
    return len(mats) / sum(map(lambda x: 1/x, mats))

def get_bedform_start_xs(offset, wavelength_cm, x_axis):
    """
    Function to calculate the start positions of all bedforms visible in the domain at a given time.

    Parameters
    ----------
    offset : float
        The offset, in cm, of the bedforms from the beginning of the domain.
    wavelength_cm : float
        The wavelength, in cm, of the bedforms.
    x_axis : array of floats
        A numpy array containing the x-coordinates of all the nodes along the x-axis of the domain.

    Returns
    -------
    bedform_start_xs : numpy array of floats
        An array of floats containing the start positions (in cm, along the x-axis) of all bedforms visible in the domain at a given time.

    """
    x_min = x_axis.min()
    x_max = x_axis.max()
    num_bedforms_before_offset = ceil((offset - x_min) / wavelength_cm)
    first_bedform_start = offset - num_bedforms_before_offset * wavelength_cm
    num_bedforms_after_offset = ceil((x_max - offset) / wavelength_cm)
    last_bedform_start = offset + (num_bedforms_after_offset - 1) * wavelength_cm
    bedform_start_xs = np.arange(first_bedform_start, last_bedform_start + 1, wavelength_cm)
    return bedform_start_xs

def get_bedform_crest_xs(offset, wavelength, x_axis, crest_offset_cm):
    """
    Function to calculate the x-locations of crests of all bedforms visible in the domain at a given time.

    Parameters
    ----------
    offset : float
        A marker giving the location in cm of the bedform that started at x = 0
    wavelength : float
        The wavelength, in cm, of a bedform.
    x_axis : numpy array of floats
        The coordinates of nodes along the x axis.
    crest_offset_cm : float
        The downstream distance in cm between the start of a bedform and its crest.

    Returns
    -------
    bedform_crest_xs:
        a numpy array of floats containing the x-locations of crests of all bedforms visible in the domain at a given time.

    """
    bedform_start_xs = get_bedform_start_xs(offset, wavelength, x_axis)
    bedform_crest_xs = bedform_start_xs + crest_offset_cm

    return bedform_crest_xs

def get_bedform_end_xs(offset, wavelength_cm, x_axis):
    """
    Function to calculate the end positions of all bedforms visible in the domain at a given time.
    (Really, the 'end position' of a bedform is the start position of the next downstream bedform).

    Parameters
    ----------
    offset : float
        The offset, in cm, of the bedforms from the beginning of the domain.
    wavelength_cm : float
        The wavelength, in cm, of the bedforms.
    x_axis : array of floats
        A numpy array containing the x-coordinates of all the nodes along the x-axis of the domain.

    Returns
    -------
    bedform_end_xs : numpy array of floats
        An array of floats containing the end positions (in cm, along the x-axis) of all bedforms visible in the domain at a given time.

    """
    bedform_start_xs = get_bedform_start_xs(offset, wavelength_cm, x_axis)
    bedform_end_xs = bedform_start_xs + wavelength_cm
    return bedform_end_xs


def get_bedform_profile_fn(offset, wavelength_cm, height_cm, crest_offset_cm, x_axis):
    """
    Generate a function that will tell us which nodes in the domain are within the bed and which are above the bed (i.e. in the surface water).

    Parameters
    ----------
    offset : float
        The offset, in cm, of the bedforms from the beginning of the domain.
    wavelength_cm : float
        The wavelength, in cm, of the bedforms.
    height_cm : float
        The trough-to-crest height, in cm, of the bedforms.
    x_axis : array of floats
        A numpy array containing the x-coordinates of all the nodes along the x-axis of the domain.

    Returns
    -------
    profile_func : TYPE
        DESCRIPTION.

    """
    start_xs_cm = get_bedform_start_xs(offset, wavelength_cm, x_axis)
    profiles = [get_bedform_profile(x, height_cm = height_cm, length_cm = wavelength_cm, crest_offset_cm = crest_offset_cm) for x in start_xs_cm]
    x = np.concatenate([profile[0] for profile in profiles])
    y = np.concatenate([profile[1] for profile in profiles])
    profile_func = interp1d(x, y)
    return profile_func

def get_bed_mask(X, Z, profile_func):
    return Z <= profile_func(X)

def get_bottom_boundary_mask(Z):
    return Z == Z.min()

def get_left_boundary_mask(X):
    return X == X.min()

def get_right_boundary_mask(X):
    return X == X.max()

def get_top_boundary_mask(Z):
    return Z == Z.max()

def get_bed_profile_mask(X, Z, masks):
    has_non_bed_node_on_right = (np.roll(masks["non_bed"], -1, 1) & masks["bed"] & ~masks["right_boundary"])
    has_non_bed_node_on_left = (np.roll(masks["non_bed"], 1, 1) & masks["bed"] & ~masks["left_boundary"])
    has_non_bed_node_above = (np.roll(masks["non_bed"], -1, 0) & masks["bed"])
    top_boundary_mask = has_non_bed_node_above | has_non_bed_node_on_right | has_non_bed_node_on_left
    return top_boundary_mask

def get_head_unknown_nodes_mask(is_bed_node, is_left_boundary, is_right_boundary, is_bed_profile):
    return is_bed_node & np.logical_not(is_left_boundary | is_right_boundary | is_bed_profile)

def get_profile_left_boundary_mask(is_bed_node, is_bed_profile):
    not_bed_node = np.logical_not(is_bed_node)
    return np.roll(not_bed_node, 1, 1) & is_bed_profile

def get_profile_right_boundary_mask(is_bed_node, is_bed_profile):
    not_bed_node = np.logical_not(is_bed_node)
    return np.roll(not_bed_node, -1, 1) & is_bed_profile

def get_touches_left_boundary_mask(is_bed_node, domain_left_boundary, profile_left_boundary):
    not_bed_node = np.logical_not(is_bed_node)
    bed_left_boundary = is_bed_node & domain_left_boundary
    touches_left_boundary = np.roll(profile_left_boundary | domain_left_boundary, 1, 1) & is_bed_node
    return touches_left_boundary

def get_touches_right_boundary_mask(is_bed_node, domain_right_boundary, profile_right_boundary):
    not_bed_node = np.logical_not(is_bed_node)
    bed_right_boundary = is_bed_node & domain_right_boundary
    touches_right_boundary = np.roll(profile_right_boundary | domain_right_boundary, -1, 1) & is_bed_node
    return touches_right_boundary

def get_touches_bed_top_boundary_mask(profile_mask, is_interior_node):
    return np.roll(profile_mask, -1, 0) & is_interior_node

def get_has_boundary_on_left_mask(is_unknown_node, domain_left_boundary, is_bed_profile):
    has_boundary_on_left = np.roll(is_bed_profile | domain_left_boundary, 1, 1) & is_unknown_node
    return has_boundary_on_left

def get_has_boundary_on_right_mask(is_unknown_node, domain_right_boundary, is_bed_profile):
    has_boundary_on_right = np.roll(is_bed_profile | domain_right_boundary, -1, 1) & is_unknown_node
    return has_boundary_on_right

def get_has_boundary_below_mask(is_bottom_boundary, is_interior_node):
    has_boundary_below = np.roll(is_bottom_boundary, 1, 0) & is_interior_node
    return has_boundary_below

def get_v_nan_right(U_x, U_z):
    v = np.sqrt(U_x**2 + U_z**2)
    v_nan_right = (np.roll(np.isnan(v), -1, 1) & ~np.isnan(v))
    return v_nan_right

def get_v_nan_left(U_x, U_z):
    v = np.sqrt(U_x**2 + U_z**2)
    v_nan_left = (np.roll(np.isnan(v), 1, 1) & ~np.isnan(v))
    return v_nan_left

def get_v_nan_above(U_x, U_z):
    v = np.sqrt(U_x**2 + U_z**2)
    v_nan_above = (np.roll(np.isnan(v), -1, 0) & ~np.isnan(v))
    return v_nan_above

def get_domain_masks(X, Z, profile_func):
    masks = {}

    masks["bed"] = get_bed_mask(X, Z, profile_func)
    masks["non_bed"] = np.logical_not(masks["bed"])
    masks["bottom_boundary"] = get_bottom_boundary_mask(Z)
    masks["left_boundary"] = get_left_boundary_mask(X)
    masks["right_boundary"] = get_right_boundary_mask(X)
    masks["bed_profile"] = get_bed_profile_mask(X, Z, masks)
    masks["profile_left_boundary"] = get_profile_left_boundary_mask(masks["bed"], masks["bed_profile"])
    masks["bed_left_boundary"] = (masks["profile_left_boundary"] | masks["left_boundary"]) & masks["bed"]
    masks["profile_right_boundary"] = get_profile_right_boundary_mask(masks["bed"], masks["bed_profile"])
    masks["bed_right_boundary"] = (masks["profile_right_boundary"] | masks["right_boundary"]) & masks["bed"]
    masks["top_boundary"] = get_top_boundary_mask(Z)
    masks["head_unknown"] = get_head_unknown_nodes_mask(masks["bed"], masks["left_boundary"], masks["right_boundary"], masks["bed_profile"])
    masks["touches_left_boundary"] = get_touches_left_boundary_mask(masks["bed"], masks["left_boundary"], masks["profile_left_boundary"])
    masks["touches_right_boundary"] = get_touches_right_boundary_mask(masks["bed"], masks["right_boundary"], masks["profile_right_boundary"])
    masks["interior"] = masks["bed"] & np.logical_not(masks["left_boundary"] | masks["right_boundary"] | masks["bed_profile"] | masks["bottom_boundary"])
    masks["touches_bed_top_boundary"] = get_touches_bed_top_boundary_mask(masks["bed_profile"], masks["interior"])
    masks["has_boundary_on_left"] = get_has_boundary_on_left_mask(masks["head_unknown"], masks["left_boundary"], masks["bed_profile"])
    masks["has_boundary_on_right"] = get_has_boundary_on_right_mask(masks["head_unknown"], masks["right_boundary"], masks["bed_profile"])
    masks["has_boundary_below"] = get_has_boundary_below_mask(masks["bottom_boundary"], masks["interior"])

    return masks

def mask_to_node_list(mask, node_order = 'horizontal'):
    """
    Function to transform a boolean mask of the bed into a list of nodes (i.e., of (i,j) index pairs) where the mask has value True.

    mask: the boolean mask of the nodes that we want to turn into a list
    node_order:
    """
    assert node_order in ('horizontal', 'vertical')

    if node_order == 'horizontal':
        output = np.asarray(np.where(mask)).T
    else:
        output = np.asarray(np.where(mask.T)[::-1]).T

    return output

def mask_node_order(mask, node_order = 'horizontal'):
    """
    Function to get the node ordering of the 'True' nodes in a boolean mask.

    The function returns a 2D array where all the 'False' nodes in the input mask are NaN, and all the 'True' nodes are numbered in order. Whether the ordering is horizontal or vertical is specified by the 'node_order' argument.
    """

    node_ij = mask_to_node_list(mask, node_order)
    i = node_ij[:,0]
    j = node_ij[:,1]

    output = np.tile(np.NaN, mask.shape)
    output[i,j] = np.arange(len(node_ij))
    return output

def update_K(K, masks, Ks_sand):
    """
    Function to update K values in the bed in light of erosion/deposition of sand along the top boundary.
    This function updates K in-place (i.e. modifies the array that is passed in, instead of making a copy of it and returning a new array).

    NOTE: for right now this function assumes that no significant amount of clay has deposited at any new nodes, i.e. that a new node starts with the default K_s value for the sand we used. (The alternative would be if, for example, the timestep used in the model was very large such that realistically a new node would have enough time to accumulate enough clay to reduce K_s by the "time" of initializing that node's K value.)
    """
    # initialize K in nodes where sand has been newly deposited (i.e. at nodes that are newly a part of the bed)
    K[masks["bed"] & np.isnan(K)] = Ks_sand

    # set K to NaN where the bed has been eroded away (i.e. at nodes that are no longer part of the bed)
    K[masks["non_bed"] & ~np.isnan(K)] = np.NaN

def low_K_layer_at_z(z_cm, thickness_cm, K, x_axis, z_axis, layer_type = 'random', reduction_factor = None):
    """

    Parameters
    ----------
    z_cm : float
        The height of the upper edge of the layer, in cm.
    thickness_cm : float
        The thickness of the layer, in cm.
    K : 2D array of floats
        Values of K at every point in the domain.
    x_axis : 1D array of floats
        The coordinates of grid nodes along the x-axis of the domain.
    z_axis : 1D array of floats
        The coordinates of grid nodes along the z-axis of the domain.
    layer_type : string, optional
        The way in which the conductivity is reduced. Either 'random', in which a random uniform distribution is sampled to determine how to reduce the layer, or 'constant', in which case the user specifies a factor between 0 and 1 by which to reduce conductivity of all nodes in the layer. The default is 'random'.
    reduction_factor : float, optional
        If 'layer_type' is 'constant', this variable specifies the constant factor (between 0 and 1) by which to multiply the conductivity of all nodes in the layer. The default is None.

    Returns
    -------
    None.

    """
    assert layer_type in ('random', 'constant')
    i_upper = np.where(z_axis == z_cm)[0][0]
    z_lower_cm = z_cm - thickness_cm
    i_lower = np.where(z_axis >= z_lower_cm)[0][0]
    layer_shape = ((i_upper - i_lower + 1), len(x_axis))
    if layer_type == 'random':
        reduction_factors = np.random.uniform(0.2, 0.5, layer_shape)
    else:
        assert type(reduction_factor) is float and 0 < reduction_factor < 1, "reduction_factor should be a float between 0 and 1, exclusive"
        reduction_factors = np.tile(reduction_factor, layer_shape)

    K[i_lower:(i_upper+1),:] *= reduction_factors

def get_initial_head_mat(K, dx, dz, masks):
    """
    This function returns the matrix that implements the discretization of the Laplacian for calculating head throughout the bed with sinusoidal head along the top of the domain and exponentially-attenuated head along the left and right boundaries.
    """
    b, c, d, e = get_head_mat_coeffs(K, dx, dz)

    unknown_nodes = mask_node_order(masks["head_unknown"])
    num_unknown_nodes = len(unknown_nodes[~np.isnan(unknown_nodes)])

    # for each of the diagonals constructed above, specify the location in the matrix of each element of the diagonal using (i, j) indexing
    i = np.arange(num_unknown_nodes)
    diag_0_indices = np.vstack([i, i]).T

    i = unknown_nodes[masks["head_unknown"] & ~masks["has_boundary_on_left"]]
    diag_minus_1_indices = np.vstack([i, i-1]).T

    i = unknown_nodes[masks["head_unknown"] & ~masks["has_boundary_on_right"]]
    diag_1_indices = np.vstack([i, i+1]).T

    i = unknown_nodes[masks["interior"]]
    j = np.roll(unknown_nodes, 1, 0)[masks["interior"]] # get the index of the node below node i in the domain
    diag_minus_n_indices = np.vstack([i, j]).T

    has_node_above = masks["head_unknown"] & np.logical_not(masks["touches_bed_top_boundary"]) # identify the nodes that have an unknown node above
    i = unknown_nodes[has_node_above]
    j = np.roll(unknown_nodes, -1, 0)[has_node_above] # get the index of the node above node i in the domain
    diag_n_indices = np.vstack([i, j]).T

    diag_0 = -(b+c+d+e)[masks["head_unknown"]]
    diag_1 = e[masks["head_unknown"] & np.logical_not(masks["has_boundary_on_right"])]
    diag_minus_1 = c[masks["head_unknown"] & np.logical_not(masks["has_boundary_on_left"])]
    diag_n_boundary = (b+d)[masks["head_unknown"] & masks["bottom_boundary"]]
    diag_n_interior = d[masks["interior"] & np.logical_not(masks["touches_bed_top_boundary"])]
    diag_n = np.concatenate([diag_n_boundary, diag_n_interior])
    diag_minus_n = b[masks["interior"]]

    # using the diagonals and index arrays constructed above, construct the head matrix as a sparse matrix
    diagonals = [diag_minus_n, diag_minus_1, diag_0, diag_1, diag_n]
    vals = np.concatenate(diagonals)
    indices = np.vstack([diag_minus_n_indices, diag_minus_1_indices, diag_0_indices, diag_1_indices, diag_n_indices])
    row = indices[:,0]
    col = indices[:,1]
    head_mat = csc_matrix((vals, (row, col)), shape = (num_unknown_nodes, num_unknown_nodes))

    return head_mat

def get_head_fn(H_m, wavelength_cm, bedform_displacement_cm, crest_offset_cm):

    bedform_head_offset = (crest_offset_cm / wavelength_cm - 0.75) * 2 * np.pi

    def head_fn(x):
        return H_m * np.sin(((x-bedform_displacement_cm) * (2*np.pi) / wavelength_cm) - bedform_head_offset)

    return head_fn

def get_optimized_head_fn(optimized_head, wavelength_cm, bedform_displacement_cm, x_axis):
    """
    Returns a function giving the optimized head at the surface of the bed as a function of x for the whole domain.

    optimized_head: the values of the optimized head function along the top of one bedform
    wavelength_cm: the wavelength of the bedforms, in cm
    bedform_displacement_cm: how far the 'chain' of bedforms has traveled since the start of the simulation.
    x_axis: the X-coordinates, in cm, of domain nodes along the X axis

    This function takes advantage of the periodic shape of the bed surface and the corresponding head distribution to propagate the head distribution downstream.
    We first extend the domain in the -x direction by prepending one bedform's worth of x coordinates and head values.
    Then we shift the extended x values in the +x direction by some offset less than wavelength_cm, to account for the current position of the bedforms.
    The value of the offset is the total distance traveled by the bedforms downstream, modulo the bedform wavelength.
    We then create an interpolation function using the shifted x values and the extended head values.
    The interpolation function is what is returned by this function.

    NOTE: this function
    """

    # gives the x values of the nodes corresponding to optimized head function, on the first bedform of the domain
    x_vals = x_axis[x_axis < wavelength_cm]

    # the assumption is that these two are the same
    # this should always be the case, considering the way head_top_BC is defined in head_optimization_for_trough_slice.py
    # (the idea is that the optimization in that script takes place only over one bedform's worth of values, which are then repeated over the whole top boundary... a shortcut that is allowable for a periodic bed shape)
    assert(len(x_vals) == len(optimized_head))

    # extend the x-axis one wavelength in the -x direction
    dx = x_axis[1] - x_axis[0]
    x_extension_cm = x_axis.min() - dx - np.arange(x_vals.max(), x_vals.min()-dx, -dx) # the extra values that we will add to the beginning of the x-axis
    extended_x_vals = np.insert(x_axis, 0, x_extension_cm)

    # extend the head distribution by one wavelength
    extended_optimized_head = np.resize(optimized_head, len(x_axis) + len(x_vals))

    x_vals_offset_cm = bedform_displacement_cm % wavelength_cm # yes, the '%' works on floating point numbers!
    extended_x_vals += x_vals_offset_cm

    head_fn = interp1d(extended_x_vals, extended_optimized_head)

    return head_fn

def head_at_surface(n, max_head, offset = 0):
    """
    Function for calculating the head along the top boundary (the surface of the bed).

    n: the number of nodes along the x or z axis of the domain
    max_head: the maximum value of the sinusoidal head distribution as given in Elliott & Brooks (Theory) 1997
    offset: the offset (in radians) for the sinusoidal head function along the top of the domain
    """
    top_boundary = max_head * np.sin(2 * np.pi * np.linspace(0, 1, n) - offset)
    return top_boundary

def impose_head_BCs(head, H_m, wavelength_cm, masks, crest_offset_cm, head_fn, X, Z, offset = 0, side_BC = "ExpDecay"):
    """
    Function to impose the head BCs on an array representing head in the domain.

    side_BC: the condition imposed at the side boundary. "ExpDecay" refers to exponential decay as in Elliott & Brooks (1997). "Zero" means setting the head to be 0 at the side boundaries (below the bed surface). "SurfaceHead" means setting the head along the whole side boundary equal to the head at the bed surface (i.e. no decay of head with depth along the side boundary). "Zero" and "SurfaceHead" are not realistic and are just used for testing.
    """
    assert side_BC in ("ExpDecay", "Zero", "SurfaceHead"), "side_BC must be set to 'ExpDecay', 'Zero', or 'SurfaceHead'"

    ### move to a separate function for getting head boundary conditions

    bed_profile_nodes = np.asarray(np.where(masks["bed_profile"])).T
    i = bed_profile_nodes[:,0]
    j = bed_profile_nodes[:,1]

    x_axis = X[0]
    head_top_BC = head_fn(x_axis[j])
    head[i, j] = head_top_BC

    is_left_BC_node = masks["bed"] & masks["left_boundary"]
    is_right_BC_node = masks["bed"] & masks["right_boundary"]

    if side_BC == "Zero":
        head[is_left_BC_node & ~masks["bed_profile"]] = 0
        head[is_right_BC_node & ~masks["bed_profile"]] = 0

    elif side_BC == "SurfaceHead":

        head[is_left_BC_node] = head[is_left_BC_node][-1]
        head[is_right_BC_node] = head[is_right_BC_node][-1]

    # side_BC equals "ExpDecay", this is our default case and it is what Elliott & Brooks (1997) prescribes
    else:
        side_BC_decay_rate = 2 * np.pi / wavelength_cm

        z_left_BC = Z[is_left_BC_node]
        depth_left_BC = z_left_BC.max() - z_left_BC[::-1]
        left_atten = np.exp(-side_BC_decay_rate * depth_left_BC)
        head_left_BC = head[is_left_BC_node][-1] * left_atten # we can do this because head[is_left_BC_node][-1] lies along the top profile of the bed, and the BC has already been set for those nodes
        head[is_left_BC_node] = head_left_BC[::-1] # reverse ordering is necessary because head_left_BC is computed from the top down, but head[is_left_BC_node] is filled in from the bottom up

        z_right_BC = Z[is_right_BC_node]
        depth_right_BC = z_right_BC.max() - z_right_BC[::-1]
        right_atten = np.exp(-side_BC_decay_rate * depth_right_BC)
        head_right_BC = head[is_right_BC_node][-1] * right_atten # we can do this because head[is_right_BC_node][-1] lies along the top profile of the bed, and the BC has already been set for those nodes
        head[is_right_BC_node] = head_right_BC[::-1]

def get_head_mat_coeffs(K, dx, dz):
    """
    Helper function to generate coefficients used in the matrices for calculating head values in the bed.
    """
    b = K / (dz**2)
    c = K / (dx**2)
    d = np.roll(K, -1, 0) / (dz**2) # d = K_(i,j+1) / delta_z**2
    e = np.roll(K, -1, 1) / (dx**2) # e = K_(i+1,j / delta_x**2

    return (b, c, d, e)

def get_head_RHS(head, K, dx, dz, masks, offset = 0):
    """
    Function to get the RHS vector for calculating the head over the bed using the Laplacian.
    """
    b, c, d, e = get_head_mat_coeffs(K, dx, dz)
    # this array stores what we will subtract from each node in generating the RHS vector
    to_subtract = np.zeros(head.shape)

    # subtract left-BC values from the unknown nodes that touch the left boundary
    unknown_touching_left_boundary = masks["head_unknown"] & masks["touches_left_boundary"]
    to_subtract[unknown_touching_left_boundary] -= (c * np.roll(head, 1, 1))[unknown_touching_left_boundary]

    # subtract right-BC values from the unknown nodes that touch the right boundary
    unknown_touching_right_boundary = masks["head_unknown"] & masks["touches_right_boundary"]
    to_subtract[unknown_touching_right_boundary] -= (e * np.roll(head, -1, 1))[unknown_touching_right_boundary]

    # subtract top-BC values from the unknown nodes that touch the top boundary
    unknown_touching_top_boundary = masks["head_unknown"] & masks["touches_bed_top_boundary"]
    to_subtract[unknown_touching_top_boundary] -= (d * np.roll(head, -1, 0))[unknown_touching_top_boundary]

    # grab only the unknown nodes from the domain as a vector
    RHS = to_subtract[masks["head_unknown"]]

    return RHS

def trans_head_mat_super_diagonal(b, d, e, masks, side, step):
    """
    Function to get the superdiagonal of the specified matrix for solving the transient groundwater flow equation using the ADI method.

    Which matrix we are calculating is specified using the 'side' and 'step' arguments.

    p = Ss / (0.5 * dt)
    r = 1 / (h**2), where h is the length/width of a cell in the domain grid
    u = 4 * r * h * Ss / dt
    masks: a dictionary containing boolean masks of the domain for different conditions. for example, masks["bed"] is the boolean mask specifying which nodes are in the stream bed
    side: 'left' or 'right': the LHS or RHS matrix for the given step of the ADI method.
    step: 0 or 1 for the first or second step of the ADI method, respectively.
    """
    boundary_node_indices = np.empty(0)
    interior_node_indices = np.empty(0)

    if side == 'left':
        if step == 0:
            boundary_node_coeff = -(b+d)
            interior_node_coeff = -d

        else: # step = 1
            boundary_node_coeff = -e
            interior_node_coeff = -e

    else: # side = 'right'
        if step == 0:
            boundary_node_coeff = e
            interior_node_coeff = e
        else:
            boundary_node_coeff = b+d
            interior_node_coeff = d

    # we are building a d/dx matrix
    if ((side == 'left' and step == 1) or (side == 'right' and step == 0)):
        head_unknown_ordered = mask_node_order(masks["head_unknown"])

        i = head_unknown_ordered[masks["head_unknown"] & masks["bottom_boundary"] & np.logical_not(masks["has_boundary_on_right"])]
        boundary_node_indices = np.vstack([i, i+1]).T.astype(np.int32)

        i = head_unknown_ordered[masks["interior"] & np.logical_not(masks["has_boundary_on_right"])]
        interior_node_indices = np.vstack([i, i+1]).T.astype(np.int32)

        boundary_node_coeffs = boundary_node_coeff[masks["head_unknown"] & masks["bottom_boundary"] & np.logical_not(masks["has_boundary_on_right"])]
        interior_node_coeffs = interior_node_coeff[masks["interior"] & np.logical_not(masks["has_boundary_on_right"])]

    # we are building a d/dz matrix
    else:
        head_unknown_ordered = mask_node_order(masks["head_unknown"], node_order = 'vertical')

        i = head_unknown_ordered[masks["head_unknown"] & masks["bottom_boundary"] & np.logical_not(masks["touches_bed_top_boundary"])]
        boundary_node_indices = np.vstack([i, i+1]).T.astype(np.int32)

        i = head_unknown_ordered[masks["interior"] & np.logical_not(masks["touches_bed_top_boundary"])]
        interior_node_indices = np.vstack([i, i+1]).T.astype(np.int32)

        # this extra sorting step is necessary because the nodes are numbered in vertical-first order, but when using masks to grab nodes from an array (in this case, when using masks to select nodes from head_unknown_ordered to generate the array i), the nodes are grabbed horizontally i.e. out of line with the order of their numbering
        interior_node_indices = np.sort(interior_node_indices, axis = 0)

        i_vert_boundary, j_vert_boundary = get_i_j_vert(masks["head_unknown"] & masks["bottom_boundary"])
        boundary_node_coeffs = boundary_node_coeff[i_vert_boundary, j_vert_boundary]

        i_vert_interior, j_vert_interior = get_i_j_vert(masks["interior"] & np.logical_not(masks["touches_bed_top_boundary"]))
        interior_node_coeffs = interior_node_coeff[i_vert_interior, j_vert_interior]

    super_diagonal = np.concatenate([boundary_node_coeffs, interior_node_coeffs])
    super_diagonal_indices = np.concatenate([boundary_node_indices, interior_node_indices])

    return super_diagonal, super_diagonal_indices

def trans_head_mat_sub_diagonal(b, c, masks, side, step):
    """
    Function to get the subdiagonal of the specified matrix for solving the transient groundwater flow equation using the ADI method.

    Which matrix we are calculating is specified using the 'side' and 'step' arguments.

    b = K / (dz**2)
    c = K / (dx**2)

    masks: a dictionary containing boolean masks of the domain for different conditions. for example, masks["bed"] is the boolean mask specifying which nodes are in the stream bed
    side: 'left' or 'right': the LHS or RHS matrix for the given step of the ADI method.
    step: 0 or 1 for the first or second step of the ADI method, respectively.
    """
    boundary_node_indices = np.empty(0)
    interior_node_indices = np.empty(0)

    if side == 'left':
        if step == 0:
            boundary_node_coeff = np.NaN
            interior_node_coeff = -b
        else: # step = 1
            boundary_node_coeff = -c
            interior_node_coeff = -c

    else: # side = 'right'
        if step == 0:
            boundary_node_coeff = c
            interior_node_coeff = c

        else: # step = 1
            boundary_node_coeff = np.NaN
            interior_node_coeff = b

    # if we are building a matrix for d/dx
    if ((side == 'left' and step == 1) or (side == 'right' and step == 0)):
        head_unknown_ordered = mask_node_order(masks["head_unknown"])

        i = head_unknown_ordered[masks["head_unknown"] & masks["bottom_boundary"] & np.logical_not(masks["has_boundary_on_left"])]
        boundary_node_indices = np.vstack([i, i-1]).T.astype(np.int32)

        i = head_unknown_ordered[masks["interior"] & np.logical_not(masks["has_boundary_on_left"])]
        interior_node_indices = np.vstack([i, i-1]).T.astype(np.int32)

        boundary_node_coeffs = boundary_node_coeff[masks["head_unknown"] & masks["bottom_boundary"] & np.logical_not(masks["has_boundary_on_left"])]
        interior_node_coeffs = interior_node_coeff[masks["interior"] & np.logical_not(masks["has_boundary_on_left"])]
    else:
        head_unknown_ordered = mask_node_order(masks["head_unknown"], node_order = 'vertical')

        i = head_unknown_ordered[masks["interior"]]
        interior_node_indices = np.vstack([i, i-1]).T.astype(np.int32)

        # this extra sorting step is necessary because the nodes are numbered in vertical-first order, but when using masks to grab nodes from an array (in this case, when using masks to select nodes from head_unknown_ordered to generate the array i), the nodes are grabbed horizontally i.e. out of line with the order of their numbering
        interior_node_indices = np.sort(interior_node_indices, axis = 0)

        i_vert, j_vert = get_i_j_vert(masks["interior"])
        interior_node_coeffs = interior_node_coeff[i_vert, j_vert]

    # start by putting the interior node coefficients and indices into the returned arrays, because the boundary node rows might not have subdiagonal coefficients in this matrix
    sub_diagonal = interior_node_coeffs
    sub_diagonal_indices = interior_node_indices

    # if there is a subdiagonal coefficient defined for the boundary node rows of this matrix, then add this coefficient and its indices into the returned arrays
    if boundary_node_coeff is not np.NaN:
        sub_diagonal = np.insert(sub_diagonal, 0, boundary_node_coeffs)
        sub_diagonal_indices = np.vstack([boundary_node_indices, sub_diagonal_indices])

    return sub_diagonal, sub_diagonal_indices

def trans_head_mat_main_diagonal(a, b, c, d, e, masks, side, step):
    """
    Function to get the main diagonal of the specified matrix for solving the transient groundwater flow equation using the ADI method.

    Which matrix we are calculating is specified using the 'side' and 'step' arguments.

    a = Ss / (dt/2)
    b = K / (dz**2)
    c = K / (dx**2)
    d = np.roll(K, -1, 0) / (dz**2) # d = K_(i,j+1) / delta_z**2
    e = np.roll(K, -1, 1) / (dx**2) # e = K_(i+1,j / delta_x**2

    masks: a dictionary containing boolean masks of the domain for different conditions. for example, masks["bed"] is the boolean mask specifying which nodes are in the stream bed
    side: 'left' or 'right': the LHS or RHS matrix for the given step of the ADI method.
    step: 0 or 1 for the first or second step of the ADI method, respectively.
    """

    interior_nodes = np.empty(0)
    boundary_nodes = np.empty(0)
    indices = np.empty(0)

    # determine the values of coefficients that will be used in the diagonal
    # in all cases, the coefficient is computed in the same way whether or not the node lies along the bottom boundary,
    # so I didn't bother with the distinction between boundary nodes and interior nodes when determining coefficient values
    if side == 'left':
        if step == 0:
            coeff = a + b + d
        else:
            coeff = a + c + e

    else:
        if step == 0:
            coeff = a - c - e
        else:
            coeff = a - b - d

    # determine if we constructing a matrix to take d/dx or d/dz?
    # if we are constructing a d/dy matrix, we'll have to deal with transposing

    # we are constructing the d/dx matrix
    if ((side == 'left' and step == 1) or (side == 'right' and step == 0)):
        diagonal = coeff[masks["head_unknown"]]

    # we are constructing the d/dz matrix
    else:
        i_vert, j_vert = get_i_j_vert(masks["head_unknown"])
        diagonal = coeff[i_vert, j_vert]

    num_unknown_nodes = len(b[masks["head_unknown"]])
    i = np.arange(num_unknown_nodes)
    indices = np.vstack([i, i]).T

    return diagonal, indices

def get_trans_head_mat(Ss, dt, K, dx, dz, masks, side, step):
    """
    Wrapper function to get the appropriate matrix for solving the transient-state groundwater flow equation (as specified by 'step' and 'side').
    """
    assert side in ('left', 'right')
    assert step in (0, 1)

    a = Ss / (dt/2)
    b, c, d, e = get_head_mat_coeffs(K, dx, dz)

    main_diagonal, main_diagonal_indices = trans_head_mat_main_diagonal(a, b, c, d, e, masks, side, step)
    sub_diagonal, sub_diagonal_indices = trans_head_mat_sub_diagonal(b, c, masks, side, step)
    super_diagonal, super_diagonal_indices = trans_head_mat_super_diagonal(b, d, e, masks, side, step)

    vals = np.concatenate([sub_diagonal, main_diagonal, super_diagonal])
    indices = np.vstack([sub_diagonal_indices, main_diagonal_indices, super_diagonal_indices])
    row = indices[:,0]
    col = indices[:,1]

    num_unknown_nodes = len(K[masks["head_unknown"]])
    mat = csc_matrix((vals, (row, col)), shape = (num_unknown_nodes, num_unknown_nodes))

    return mat

def get_i_j_vert(mask):
    """
    Sometimes, we need index an array using a boolean mask, but traverse the nodes in vertical-first order.
    This is a helper function to precompute the indices for the desired nodes in the desired order.
    Returns:
        i_vert: an array of i indices
        j_vert: an array of j indices

    Then, indexing the target array this way gives the desired nodes in the desired order:
        target_arr[i_vert, j_vert]

    """
    head_unknown_nodes_V = mask_to_node_list(mask, node_order = 'vertical')
    i_vert = head_unknown_nodes_V[:,0]
    j_vert = head_unknown_nodes_V[:,1]

    return i_vert, j_vert

def get_trans_head_RHS_vec(head, RHS_head_mat, masks, Ss, dt, K, dx, dz, step):
    _, c, d, e = get_head_mat_coeffs(K, dx, dz)

    output = np.zeros(head.shape)

    i_vert, j_vert = get_i_j_vert(masks["head_unknown"])

    if step == 0:
        output[masks["head_unknown"]] = RHS_head_mat.dot(head[masks["head_unknown"]])
        output[masks["has_boundary_on_left"]] += c[masks["has_boundary_on_left"]] * np.roll(head, 1, 1)[masks["has_boundary_on_left"]]
        output[masks["has_boundary_on_right"]] += e[masks["has_boundary_on_right"]] * np.roll(head, -1, 1)[masks["has_boundary_on_right"]]
        output[masks["touches_bed_top_boundary"]] += d[masks["touches_bed_top_boundary"]] * np.roll(head, -1, 0)[masks["touches_bed_top_boundary"]]

        RHS_vec = output[i_vert, j_vert]

    else:
        output[i_vert, j_vert] = RHS_head_mat.dot(head[i_vert, j_vert])
        output[masks["touches_bed_top_boundary"]] += d[masks["touches_bed_top_boundary"]] * np.roll(head, -1, 0)[masks["touches_bed_top_boundary"]]
        output[masks["has_boundary_on_left"]] += c[masks["has_boundary_on_left"]] * np.roll(head, 1, 1)[masks["has_boundary_on_left"]]
        output[masks["has_boundary_on_right"]] += e[masks["has_boundary_on_right"]] * np.roll(head, -1, 1)[masks["has_boundary_on_right"]]

        RHS_vec = output[masks["head_unknown"]]

    return RHS_vec

def get_coord_grid(x_min, x_max, n):
    """
    Function to get coordinate grid for making contour plots over the domain.
    """
    axis = np.linspace(x_min, x_max, n)
    x, y = np.meshgrid(axis, axis)

    return x, y

def cbar_tick_label_to_float(label):
    """
    Function to handle some weird hyphen-like character when converting colorbar label texts to floats. This is a helper function to new_label_from_old().
    """
    target = label.get_text()
    if ord(target[0]) == 8722:
        target = '-' + target[1:]
    return float(target)


def new_label_from_old(old_label):
    """
    Function to format colorbar labels for head_contour_plot().
    """
    flt_val = cbar_tick_label_to_float(old_label)
    flt_val *= 1000
    new_label = '{:3.2f}'.format(flt_val)
    return new_label

def head_contour_plot(head, H_m, wavelength_cm, crest_offset_cm, offset, X, Z):
    """
    Function to generate a contour plot of head throughout the bed.

    head: the values of head throughout the bed
    x_min: the lower limit of the x axis of the model
    x_max: the upper limit of the x axis of the model
    n: the number of nodes along the x or y axis
    """
    x_axis = X[0] # get the coordinates of the x-axis from the X that came from np.meshgrid()
    head_fn = get_head_fn(H_m, wavelength_cm, offset, crest_offset_cm)
    profile_func = get_bedform_profile_fn(offset, wavelength_cm, height_cm, crest_offset_cm, x_axis)
    x_profile = np.linspace(x_axis.min(), x_axis.max(), 1000)
    y = profile_func(x_profile)

    fig, ax_list = get_domain_plot_template()
    ax1, ax2 = ax_list

    ax1.set_title("Imposed Head (cm)")
    head_top_BC = head_fn(x_axis)
    ax1.plot(x_axis, head_top_BC)
    ax1.hlines(0, x_axis.min(), x_axis.max(), linestyle = '--', color = 'gray')

    ax2.plot(x_profile, y, color = colors[0], label = 'Bed Profile')
    cs = ax2.contourf(X, Z, head, 15)
    cbar = fig.colorbar(cs)
    cbar_label = plt.text(0.87, 0.35, "Head (cm)", transform = fig.transFigure, rotation = 270)

    ax1_bbox = ax1.get_position()
    ax2_bbox = ax2.get_position()
    ax1_new_bbox = Bbox([[ax1_bbox.xmin, ax1_bbox.ymin], [ax2_bbox.xmax, ax1_bbox.ymax]])
    ax1.set_position(ax1_new_bbox)

    plt.title("Head in the Bed")
    plt.xlabel("Distance Downstream (cm)")
    plt.ylabel("Depth (cm)")

    arrow_x_loc = (ax1_new_bbox.xmax + ax1_new_bbox.xmin) / 2
    flow_text = plt.text(arrow_x_loc, 0.9, 'Flow Direction', ha = 'center', fontsize = 14, bbox = dict(boxstyle='rarrow,pad=0.2', fc = 'xkcd:lightblue', ec = 'black', lw = 1), transform = plt.gcf().transFigure)

    cb_ax = fig.axes[2]
    cb_labels = cb_ax.get_yticklabels()
    new_labels = [new_label_from_old(label) for label in cb_labels]
    cb_ax.set_yticklabels(new_labels)
    offset_text = cb_ax.text(2, 1.05, r"x $10^{-3}$", bbox = {"linewidth": 1, "fill": False}, transform = cb_ax.transAxes)

    return fig

def plot_head(head, head_imp, x_min, x_max, n):
    """
    Function to plot the head throughout the bed.
    """
    fig = head_contour_plot(head, head_imp, x_min, x_max, n)

    bed_plot = fig.axes[1]
    bed_plot.set_title("Head Throughout the Bed")
    bed_plot.set_xlabel("X (Distance Downstream)")
    bed_plot.set_ylabel("Z")

    return fig

def plot_bed_flow(head, Ux, Uz, K, H_m, wavelength_cm, offset, X, Z, masks):
    """
    Function to plot flow paths in the bed, superimposed on a contour plot of head throughout the bed.

    """
    fig = head_contour_plot(head, H_m, wavelength_cm, offset, X, Z)
    flow_paths = plt.quiver(X[masks["head_unknown"]], Z[masks["head_unknown"]],  U_x[masks["head_unknown"]], U_z[masks["head_unknown"]], scale = 2.5*(10**-12), units = 'xy')

    plt.title("Flow Lines In The Bed")

    return fig

def get_dh_dx_mat(head, masks, h):
    """
    Function to get the matrix used to calculate dh/dx throughout the bed.

    head: a 2D array specifying head values at all the nodes in the domain
    masks: a dict of masks as created by get_domain_masks()
    h: the distance between nodes in the x and z directions (assumes that the spacing for x is the same as the spacing for z)
    """
    i = np.where(~masks["has_boundary_on_left"][masks["head_unknown"]])[0]
    sub_diagonal_indices = np.vstack([i, i-1]).T

    i = np.where(~masks["has_boundary_on_right"][masks["head_unknown"]])[0]
    super_diagonal_indices = np.vstack([i, i+1]).T

    sub_diagonal_values = np.repeat(-1, len(sub_diagonal_indices))
    super_diagonal_values = np.repeat(1, len(sub_diagonal_indices))

    indices = np.vstack((sub_diagonal_indices, super_diagonal_indices))
    values = np.concatenate([sub_diagonal_values, super_diagonal_values])
    values = values / (2*h)

    row = indices[:,0]
    col = indices[:,1]

    num_unknown_nodes = len(np.where(masks["head_unknown"])[0])
    output = csc_matrix((values, (row, col)), shape = (num_unknown_nodes, num_unknown_nodes))

    return output

def get_dh_dx_BC_additions(head, masks, h):
    """
    Compute the values that needed to be added to each node due to boundary conditions when calculating dh_dx (head gradients in the x-direction).

    head: a 2D array specifying head values at all the nodes in the domain
    masks: a dict of masks as created by get_domain_masks()
    h: the distance between nodes in the x and z directions (assumes that the spacing for x is the same as the spacing for z)
    """
    output = np.zeros(head.shape)
    output[masks["non_bed"]] = np.NaN
    output[masks["has_boundary_on_left"]] -= np.roll(head, 1, 1)[masks["has_boundary_on_left"]]
    output[masks["has_boundary_on_right"]] += np.roll(head, -1, 1)[masks["has_boundary_on_right"]]
    output /= (2*h)
    return output

def get_dh_dx(head, masks, dh_dx_mat, h):
    dh_dx = np.tile(np.NaN, head.shape)
    dh_dx_BC_additions = get_dh_dx_BC_additions(head, masks, h)
    dh_dx[masks["head_unknown"]] = dh_dx_mat.dot(head[masks["head_unknown"]]) + dh_dx_BC_additions[masks["head_unknown"]]
    return dh_dx

def get_dh_dz_mat(masks, h):
    """
    Function to get the matrix used to calculate dh/dz throughout the bed.
    Note that the returned matrix assumes vertical ordering of nodes.

    masks: a dict of masks as created by get_domain_masks()
    h: the distance between nodes in the x and z directions (assumes that the spacing for x is the same as the spacing for z)
    """
    interior_nodes = mask_to_node_list(masks["interior"], node_order = 'vertical')

    # no need to use vertical node ordering for these two sets of nodes,
    # since we just want to know if a given node is in the list or not
    has_boundary_below_nodes = mask_to_node_list(masks["has_boundary_below"])
    touches_bed_top_boundary_nodes = mask_to_node_list(masks["touches_bed_top_boundary"])

    i = np.arange(len(interior_nodes))
    main_diagonal_indices = np.vstack([i, i]).T

    sub_diagonal_nodes_mask = (masks["interior"] & ~masks["has_boundary_below"])
    i = interior_nodes[:,0]
    j = interior_nodes[:,1]
    sub_diagonal_nodes = sub_diagonal_nodes_mask[i,j]

    i = np.where(sub_diagonal_nodes)[0]
    sub_diagonal_indices = np.vstack([i,i-1]).T

    main_diagonal_values = np.repeat(1, len(main_diagonal_indices))
    sub_diagonal_values = np.repeat(-1, len(sub_diagonal_indices))

    indices = np.vstack((main_diagonal_indices, sub_diagonal_indices))
    values = np.concatenate([main_diagonal_values, sub_diagonal_values])
    values = values / h

    row = indices[:,0]
    col = indices[:,1]

    num_interior_nodes = len(interior_nodes)
    output = csc_matrix((values, (row, col)), shape = (num_interior_nodes, num_interior_nodes))

    return output

def get_dh_dz_BC_additions(head, masks, h):
    output = np.zeros(head.shape)
    output[masks["non_bed"]] = np.NaN
    output[masks["has_boundary_below"]] -= np.roll(head, 1, 0)[masks["has_boundary_below"]]
    output /= h
    return output

def get_dh_dz(head, masks, dh_dz_mat, h):
    dh_dz = np.tile(np.NaN, head.shape)

    # get indices of interior nodes, traversing the nodes in a vertical-first order
    i_vert, j_vert = get_i_j_vert(masks["interior"])

    dh_dz[i_vert, j_vert] = dh_dz_mat.dot(head[i_vert, j_vert])
    dh_dz_BC_additions = get_dh_dz_BC_additions(head, masks, h)
    dh_dz += dh_dz_BC_additions

    # this is part of the model definition: no flux across the bottom boundary
    dh_dz[masks["bottom_boundary"]] = 0

    return dh_dz

def show_nodes(xmin, xmax, n):
    """
    Helper function for debugging.

    Superimposes node locations on top of an existing plot like the one produced by plot_C().

    xmin: the minimum x value of the domain.
    xmax: the maximum x value of the domain.
    n: the number of nodes along one axis of the domain.
    """
    x, y = get_coord_grid(xmin, xmax, n)
    plt.scatter(x, y, color = 'red', s = 9)

def get_consts(D, dt, h, l):
    """
    Calculate constants that make the construction of the matrices easier to read and understand.

    D: diffusion constant
    dt: size of the timestep (i.e., delta t)
    h: the distance between nodes along the domain grid
    l: the decay constant governing how much of the mobile clay gets turned into immobile clay with each timestep
    """

    a = (D * dt) / (2 * h**2)
    b = dt / (2*h)
    g = dt / (2*h)
    k = l * dt / 4

    return (a, b, g, k)

def get_LHS_row_coeffs(i, j, D, dt, h, Ux, Uz, l, n, step):
    """
    Helper function to get the set of three coefficients than can appear in a given row of a LHS matrix.

    i: the row number in the domain of the node whose row we are constructing
    j: the column number in the domain of the node whose row we are constructing
    D: diffusion constant
    dt: size of the timestep (i.e., delta t)
    h: the distance between nodes along the domain grid
    Ux: the matrix of velocities in the x direction
    Uz: the matrix of velocities in the z direction
    l: the decay constant governing how much of the mobile clay gets turned into immobile clay with each timestep
    n: the number of nodes along the X or Z axis in the domain grid
    step: the step (first or second) of the split-operator procedure that we are on. If it's the step from t to t+(1/2), step = 0. Else step = 1.

    """

    a, b, g, k = get_consts(D, dt, h, l)
    b = b * Ux
    g = g * Uz.T

    # find the three coefficients that can appear in a given row of the matrix
    if step == 0:
        if g[i, j] < 0:
            coeffs = np.asarray([-a, (1 + (2*a) - g[i,j] + k), -a + g[i,j]])
        else:
            coeffs = np.asarray([(-a - g[i,j]), (1 + (2*a) + g[i,j] + k), -a])
    else:
        if b[i,j] >= 0:
            coeffs = np.asarray([-a - b[i,j], 1 + (2*a) + k + b[i,j], -a])
        else:
            coeffs = np.asarray([-a, 1 + (2*a) + k - b[i,j], -a + b[i,j]])

    return coeffs

def get_RHS_row_coeffs(i, j, D, dt, h, Ux, Uz, l, n, step):
    """
    Helper function to get the set of three coefficients than can appear in a given row of a RHS matrix.

    i: the row number in the domain of the node whose row we are constructing
    j: the column number in the domain of the node whose row we are constructing
    D: diffusion constant
    dt: size of the timestep (i.e., delta t)
    h: the distance between nodes along the domain grid
    Ux: the matrix of velocities in the x direction
    Uz: the matrix of velocities in the z direction
    l: the decay constant governing how much of the mobile clay gets turned into immobile clay with each timestep
    n: the number of nodes along the X or Z axis in the domain grid
    step: the step (first or second) of the split-operator procedure that we are on. If it's the step from t to t+(1/2), step = 0. Else step = 1.

    """
    a, b, g, k = get_consts(D, dt, h, l)
    b = b * Ux
    g = g * Uz.T

    if step == 0:
        if b[i,j] >= 0:
            coeffs = np.asarray([a + b[i,j], 1 - (2*a) - k - b[i,j], a])
        else:
            coeffs = np.asarray([a, 1 - (2*a) - k + b[i,j], a - b[i,j]])
    else:
        if g[i,j] < 0:
            coeffs = np.asarray([a, 1 - (2*a) + g[i,j] - k, a - g[i,j]])
        else:
            coeffs = np.asarray([a + g[i,j], 1 - (2*a) - g[i,j] - k, a])

    return coeffs

def get_coeffs_for_top_BC(i, j, D, dt, h, Ux, Uz, l, n):
    """
    Helper function to get coefficients by which to multiply top-boundary condition values. Called in get_RHS_vec().

    i: the row number in the domain of the node whose row we are constructing
    j: the column number in the domain of the node whose row we are constructing
    D: diffusion constant
    dt: size of the timestep (i.e., delta t)
    h: the distance between nodes along the domain grid
    Ux: the matrix of velocities in the x direction
    Uz: the matrix of velocities in the z direction
    l: the decay constant governing how much of the mobile clay gets turned into immobile clay with each timestep
    n: the number of nodes along the X or Z axis in the domain grid
    """

    a, b, g, k = get_consts(D, dt, h, l)
    b = b * Ux
    g = g * Uz

    if g[i,j] < 0:
            coeffs = np.asarray([a, 1 - (2*a) + g[i,j] - k, a - g[i,j]])
    else:
            coeffs = np.asarray([a + g[i,j], 1 - (2*a) - g[i,j] - k, a])

    return coeffs

def mat_row_from_coeffs(ind, n, coeffs):
    """
    Helper function to construct a matrix row from a given set of three row coefficients

    ind: the index of the node whose row we are constructing. in theory ind can range from 0 to (n**2)-1 but in practice it will range from (n+1) to (n-1)**2 - 1, since we only deal with non-boundary nodes.
    n: the number of nodes along one axis of the domain
    coeffs: the set of three row coefficients
    """

    # transform the absolute index of the node into typical (i,j) coordinates
    i = ind % n
    j = int(ind / n)

    assert i >= 1 and i <= (n-2), "Error: i is not a node for which we need to construct a row. i = " + str(i) + ", n = " + str(n) + ", ind = " + str(ind)

    # construct the matrix, using the calculated row coefficients
    if i == 1:
        left_part = coeffs[1:]
        right_part = np.zeros(n-4)
        row = np.concatenate([left_part, right_part])
    elif i == (n-2):
        left_part = np.zeros(n-4)
        right_part = coeffs[:2]
        row = np.concatenate([left_part, right_part])
    else:
        left_part = np.zeros(i-2)
        center_part = coeffs
        right_part = np.zeros(n-3-i)
        row = np.concatenate([left_part, center_part, right_part])

    left_zeros = np.zeros((j-1)*(n-2))
    right_zeros = np.zeros((n-2) * ((n-2)-j))

    row = np.concatenate([left_zeros, row, right_zeros])

    return row

def get_domain_plot_template():
    """
    Generates a figure with correctly shaped subplots and an arrow indicating flow direction.

    Useful for plotting various quantities over the domain.
    """
    fig, ax_list = plt.subplots(2,1, sharex = True, figsize = (10,6), gridspec_kw={"height_ratios": (0.2,1), "top" : 0.75, "hspace": 0.4})
    return (fig, ax_list)

def format_hrs_str(hrs):
    """
    Helper function to format the string displaying a number of hours in the animation of clay concentrations in the bed.

    hrs: a float, the number of hours elapsed
    """
    output = "{:5.2f}".format(hrs)
    return output

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1],
                 [2,3],
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def get_bedform_profile(start_x_cm, start_y_cm = 20, length_cm = 25, height_cm = 5, crest_offset_cm = 22):

    crest_x_cm = start_x_cm + crest_offset_cm

    nPoints = 4
#    points = np.random.rand(nPoints,2)*200
    points = np.asarray([[start_x_cm, start_y_cm],
                         [start_x_cm + 0.6 * length_cm, start_y_cm],
                         [start_x_cm + (11/25) * length_cm, start_y_cm + height_cm],
                         [crest_x_cm, start_y_cm + height_cm]])

    xpoints = [p[0] for p in points]
    ypoints = [p[1] for p in points]

    xvals, yvals = bezier_curve(points, nTimes=1000)

    xvals = np.flip(xvals)
    yvals = np.flip(yvals)

    t = np.linspace(0, 10, 11)

    end_x_cm = start_x_cm + length_cm
    slope = -height_cm / (end_x_cm - crest_x_cm)
    lee_x_cm = crest_x_cm + ((end_x_cm - crest_x_cm) / (len(t)-1)) * t
    lee_y_cm = (start_y_cm + height_cm) + slope * (lee_x_cm - crest_x_cm)

    xvals = np.concatenate([xvals, lee_x_cm])
    yvals = np.concatenate([yvals, lee_y_cm])

    return xvals, yvals

def plot_head_vs_dh_dz(x, head, dh_dz, Z, x_axis):
    """
    Function to plot head and dh_dz vs. depth in the bed at a given value of x (cm).
    For use in debugging.
    """
    j = np.where(x_axis == x)[0]
    fig = plt.figure()
    plt.plot(head[:,j], Z[:,j], marker = 'o', label = 'Head')
    ax1 = plt.gca(); ax2 = ax1.twiny();
    ax2.plot(dh_dz[:,j], Z[:,j], marker = 's', color = 'orange', label = r"$\frac{\partial h}{\partial z}$")
    ax1.set_xlabel("Head")
    ax2.set_xlabel(r"$\frac{\partial h}{\partial z}$")
    ax1.set_ylabel("Z (cm)")
    plt.legend([ax1.lines[0], ax2.lines[0]], ['Head', dh_dz_str],  loc = 'best')
    return fig

def plot_K_vs_dh_dz(x, K, dh_dz, Z, x_axis):
    """
    Function to plot K and dh_dz vs. depth in the bed at a given value of x (cm).
    For use in debugging.
    """
    j = np.where(x_axis == x)[0]
    fig = plt.figure()
    plt.plot(K[:,j], Z[:,j], marker = '^', color = 'green', label = 'K')
    ax1 = plt.gca(); ax2 = ax1.twiny();
    ax2.plot(dh_dz[:,j], Z[:,j], marker = 's', color = 'orange', label = r"$\frac{\partial h}{\partial z}$")
    ax1.set_xlabel("K")
    ax2.set_xlabel(r"$\frac{\partial h}{\partial z}$")
    ax1.set_ylabel("Z (cm)")
    plt.legend([ax1.lines[0], ax2.lines[0]], ['K', dh_dz_str],  loc = 'best')
    return fig

def plot_head_vs_K(x, head, K, Z, x_axis):
    """
    Function to plot head and K vs. depth in the bed at a given value of x (cm).
    For use in debugging.
    """
    j = np.where(x_axis == x)[0]
    fig = plt.figure()
    plt.plot(head[:,j], Z[:,j], marker = 'o', label = 'Head')
    ax1 = plt.gca(); ax2 = ax1.twiny();
    ax2.plot(K[:,j], Z[:,j], marker = 's', color = 'orange', label = "K")
    ax1.set_xlabel("Head")
    ax2.set_xlabel("K")
    ax1.set_ylabel("Z (cm)")
    plt.legend([ax1.lines[0], ax2.lines[0]], ['head', 'K'],  loc = 'best')
    return fig

def plot_head_at_x(x, head, Z, x_axis):
    j = np.where(x_axis == x)[0]
    fig = plt.figure()
    plt.plot(head[:,j], Z[:,j], marker = 'o', label = 'Head')
    ax1 = plt.gca()
    ax1.set_xlabel("Head")
    ax1.set_ylabel("Z (cm)")
    plt.legend(loc = 'best')
    return fig

### Begin functions for microplastics particle-tracking simulation ###

def get_particle_init_pos(num_particles):
    x = np.random.uniform(10, 14, num_particles)
    z = np.repeat(19.5, num_particles)
    pos = np.vstack([x, z]).T
    return pos

def get_particle_init_pos_line(start_pt, end_pt, num_particles):
    slope = (end_pt[1] - end_pt[0]) / (start_pt[1] - start_pt[0])
    x = np.random.uniform(start_pt[0], end_pt[0], num_particles)
    z = start_pt[1] + slope*(x - start_pt[0])
    pos = np.vstack([x,z]).T
    return pos

def get_particle_U(pos, U_x, U_z, X, Z, masks):
    """
    Function to calculate the velocity vector at each particle's location.

    Returns a nx2 array, where in is the number of particles.
    The ith row corresponds to the ith particle.
    Column 0 is the x-component of each particle's velocity, and column 1 is the z component.
    """
    pos_z_min = pos["Z_cm"].min()
    domain_z_min = Z[Z < pos_z_min].max()
    known_locs = masks["head_unknown"] & (Z >= domain_z_min)

    node_xz_coords = np.vstack([X[known_locs], Z[known_locs]]).T
    particle_Ux = griddata(node_xz_coords, U_x[known_locs], pos)
    particle_Uz = griddata(node_xz_coords, U_z[known_locs], pos)
    particle_U = np.vstack([particle_Ux, particle_Uz]).T
    return particle_U

def update_dep(dep, displacement, l):
    """
    Update the array showing which particles have deposited and which haven't.

    dep: an array of Booleans. Each element represents a particle. 'True' indicates that that particle has deposited, 'False' indicates it is mobile.
    displacement: the x- and z- components of each particle's displacment during the current timestep. This is an nx2 array of floats, where n is the number of particles.
    l: the filtration coefficient (appears as lambda_f in the literature). this is a float, with units 1/cm. It expresses the percentage of clay particles that deposit per centimeter traveled in the bed.
    """
    num_particles = dep.shape[0]
    ds = np.apply_along_axis(norm, 1, displacement)
    dep_prob = 1 - np.exp(-l * ds)
    rand_num = np.random.rand(num_particles)
    dep = ((rand_num < dep_prob) | (dep == True))
    return dep

def get_U_T(particle_U):
    """


    Parameters
    ----------
    particle_U : an nx2 array of floats representing particle velocities, as generated by get_particle_U

    Returns
    -------
    U_T, an nx2 array of floats, where each row is a vector orthogonal to a given particle's current velocity. Used for computing transverse dispersion.

    """
    U_T = particle_U[:, ::-1].copy()
    U_T[:,0] *= -1
    return U_T

def get_displacement(particle_U, dt, a_L, a_T):
    """
    Function to calculate displacement for particle tracking method. Includes advective displacement as well as dispersion.

    particle_U: an nx2 array of floats representing particle velocities as generated by get_particle_U()
    dt: the amount of time (in seconds) that passes between each timestep
    a_L: longitudinal dispersivity, in units of cm
    a_T: transverse dispersivity, in units of cm
    """
    U_T = get_U_T(particle_U)
    v = np.apply_along_axis(lambda x: norm(x, 2), 1, particle_U)
    D_L = v * a_L
    D_T = v * a_T

    rand_num = np.random.normal(size = len(v))
    coeff_L = rand_num * np.sqrt(2 * D_L * dt)
    dispersion_L = np.asarray([a * v for a, v in zip(coeff_L, particle_U)])
    coeff_T = rand_num * np.sqrt(2 * D_T * dt)
    dispersion_T = np.asarray([a * v for a, v in zip(coeff_T, U_T)])

    advection = dt * particle_U
    displacement = advection + dispersion_L + dispersion_T

    return displacement

def in_lee_range(X_grid, lee_range):
    """
    Find the the nodes of the X grid where the x coordinate is in the lee side of a bedform.
    Note: the use of >= for the beginning of the range but < for the end is intentional. The idea is to grab nodes that have a boundary node to their right.

    Parameters
    ----------
    X_grid : a 2D numpy array of X coordinates of all nodes in the domain. (in run_model.py, this is simply the X array.)
    lee_range : a numpy array of shape (1,2), specifying the beginning and ending x coordinates of the lee face of a given bedform.

    Returns
    -------
    A 2D boolean mask over X_grid, specifying which nodes do or don't have an x coordinate that's in the lee side of a bedform.

    """
    return (X_grid >= lee_range[0]) & (X_grid < lee_range[1])

def in_stoss_range(X_grid, stoss_range):
    return (X_grid >= stoss_range[0]) & (X_grid < stoss_range[1])

def get_bedform_stoss_ranges(offset, wavelength_cm, x_axis, crest_offset_cm, exclude_margins = True, head_margin_cm = 5):
    bedform_start_xs = get_bedform_start_xs(offset, wavelength_cm, x_axis)
    bedform_crest_xs = get_bedform_crest_xs(offset, wavelength_cm, x_axis, crest_offset_cm)
    bedform_stoss_ranges = np.vstack([bedform_start_xs, bedform_crest_xs]).T

    # exclude the left and right margins of the domain from particle tracking calculations due to anomalous behavior of head in those regions
    bedform_stoss_ranges = np.asarray([x for x in map(lambda a: exclude_margin(a, head_margin_cm, x_axis), bedform_stoss_ranges) if x[0] < x[1]])

    return bedform_stoss_ranges

def get_bedform_lee_ranges(offset, wavelength_cm, x_axis, crest_offset_cm, exclude_margins = True, head_margin_cm = 5):
    bedform_crest_xs = get_bedform_crest_xs(offset, wavelength_cm, x_axis, crest_offset_cm)
    bedform_end_xs = get_bedform_end_xs(offset, wavelength_cm, x_axis)
    bedform_lee_ranges = np.vstack([bedform_crest_xs, bedform_end_xs]).T

    # exclude the left and right margins of the domain from particle tracking calculations due to anomalous behavior of head in those regions
    bedform_lee_ranges = np.asarray([x for x in map(lambda a: exclude_margin(a, head_margin_cm, x_axis), bedform_lee_ranges) if x[0] < x[1]])

    return bedform_lee_ranges

def get_v_known_lee_masks(offset, wavelength_cm, x_axis, crest_offset_cm, U_x, U_z):
    bedform_crest_xs = get_bedform_crest_xs(offset, wavelength_cm, x_axis, crest_offset_cm)
    bedform_end_xs = get_bedform_end_xs(offset, wavelength_cm, x_axis)
    bedform_lee_ranges = np.vstack([bedform_crest_xs, bedform_end_xs]).T
    v_nan_right = get_v_nan_right(U_x, U_z)
    v_known_lee_masks = [in_lee_range(X, lee_range) & v_nan_right for lee_range in bedform_lee_ranges]
    return v_known_lee_masks

def get_v_known_stoss_masks(offset, wavelength_cm, x_axis, crest_offset_cm, U_x, U_z):
    bedform_start_xs = get_bedform_start_xs(offset, wavelength_cm, x_axis)
    bedform_crest_xs = get_bedform_crest_xs(offset, wavelength_cm, x_axis, crest_offset_cm)
    bedform_stoss_ranges = np.vstack([bedform_start_xs, bedform_crest_xs]).T
    v_nan_above = get_v_nan_above(U_x, U_z)
    v_nan_left = get_v_nan_left(U_z, U_z)
    v_known_stoss_masks = [in_stoss_range(X, stoss_range) & v_nan_above for stoss_range in bedform_stoss_ranges]
    return v_known_stoss_masks

def get_tangent_at_surface_nodes(interval):
    """

    Parameters
    ----------
    interval : a numpy array of floats, of length 2
        Specifies the start and end points of the interval along the x-axis for a given stoss or lee face of a bedform

    Returns
    -------
    a numpy array of floats of shape Mx2, where M is the number of surface nodes in the interval.
    The mth row gives the vector tangent to the bed surface at the mth surface node, in the downstream direction.

    """
    x_stoss = X[stoss_mask]
    x_slope = np.linspace(stoss_range.min(), stoss_range.max(), 1000)
    z_slope = profile_func(x_slope)
    dz = z_slope[1:] - z_slope[:-1]
    dx = x_slope[1:] - x_slope[:-1]
    f_dz = interp1d(x_slope[:-1], dz)
    dz_stoss = f_dz(x_stoss)
    incr = x_slope[1] - x_slope[0]
    dx_stoss = np.repeat(incr, len(dz_stoss))
    tangents = np.vstack([dx_stoss, dz_stoss]).T

def get_stoss_mask(stoss_range, U_x, U_z, X):
    v_nan_above = get_v_nan_above(U_x, U_z)
    stoss_mask = in_stoss_range(X, stoss_range) & v_nan_above
    return stoss_mask

def get_dz_interp_incr(stoss_range, U_x, U_z, X):
    stoss_mask = get_stoss_mask(stoss_range, U_x, U_z, X)
    x_stoss = X[stoss_mask]
    interp_incr = (stoss_range[1] - x_stoss[-1]) - eps
    return interp_incr

def get_x_for_interp(stoss_range, U_x, U_z, X, num_pts = 1000):
    stoss_mask = get_stoss_mask(stoss_range, U_x, U_z, X)
    incr = (stoss_range.max() - stoss_range.min()) / (num_pts - 1)
    new_max = stoss_range.max() + incr
    x_slope = np.linspace(stoss_range.min(), new_max, num_pts+1)
    return x_slope

def get_dx_stoss(stoss_range, U_x, U_z, X):
    stoss_mask = get_stoss_mask(stoss_range, U_x, U_z, X)
    x_stoss = X[stoss_mask]
    x_interp = get_x_for_interp(stoss_range, U_x, U_z, X)
    x_interp_incr = x_interp[1] - x_interp[0] # assumes that the x-increment between interpolation points is constant
    dx = np.repeat(x_interp_incr, len(x_stoss))
    return dx

def get_dz_fn(stoss_range, U_x, U_z, X, profile_func):
    """
    Function to generate an interpolation function to find dz at a given point on the stoss side of a bedform.

    """
    # interp_incr = get_dz_interp_incr(stoss_range, U_x, U_z, X)
    # num_pts_for_interp = ceil((stoss_range.max() - stoss_range.min()) / interp_incr) + 1
    x_slope = get_x_for_interp(stoss_range, U_x, U_z, X)
    z_slope = profile_func(x_slope)
    dz = z_slope[1:] - z_slope[:-1]
    dx = x_slope[1:] - x_slope[:-1]
    f_dz = interp1d(x_slope[:-1], dz)

    return f_dz

def get_flow_is_inward(stoss_range, U_x, U_z, X, profile_func):
    stoss_mask = get_stoss_mask(stoss_range, U_x, U_z, X)
    x_stoss = X[stoss_mask]
    f_dz = get_dz_fn(stoss_range, U_x, U_z, X, profile_func)
    dz_stoss = f_dz(x_stoss)
    dx_stoss = get_dx_stoss(stoss_range, U_x, U_z, X)
    tangents = np.vstack([dx_stoss, dz_stoss]).T
    U_stoss = np.vstack([U_x[stoss_mask], U_z[stoss_mask]]).T
    dets = np.asarray([det(np.vstack([U_stoss[i], tangents[i]])) for i in range(len(tangents))])
    flow_inward = dets > 0

    return flow_inward

def get_stoss_normal_flux(stoss_range, profile_func, U_x, U_z, X):
    # get the porewater velocity at the nodes we're interested in.
    # nodes we're interested in: nodes at the upper boundary of the region where we know porewater velocity and on the stoss side of this specific bedform
    stoss_mask = get_stoss_mask(stoss_range, U_x, U_z, X)
    U_stoss = np.vstack([U_x[stoss_mask], U_z[stoss_mask]]).T

    x_stoss = X[stoss_mask]
    f_dz = get_dz_fn(stoss_range, U_x, U_z, X, profile_func)
    dz_stoss = f_dz(x_stoss)
    dx_stoss = get_dx_stoss(stoss_range, U_x, U_z, X)
    tangents = np.vstack([dx_stoss, dz_stoss]).T

    # get the vector tangent to the bedform surface at the x-coordinate of every node we're interested in.
    normal_inward = np.vstack([tangents[:,1], -tangents[:,0]]).T

    # get the normal-inward component of the stoss-side flux entering the bedform
    # note: this code does not exclude nodes where the flux is out of the the bed; that is left for elsewhere
    scalar_projection = np.asarray([U_stoss[i].dot(normal_inward[i]) for i in range(len(U_stoss))])
    normal_inward_mag = np.apply_along_axis(lambda x: norm(x), 1, normal_inward)
    scale_factor = scalar_projection / normal_inward_mag**2
    vector_projection = np.asarray([normal_inward[i] * scale_factor[i] for i in range(len(scale_factor))])
    stoss_normal_flux = vector_projection

    return stoss_normal_flux

def get_new_particles_stoss(stoss_range, profile_func, U_x, U_z, surface_MP_conc_ppL, X, x_axis, Z, dt, porosity = 0.33, flume_width_cm = 30):
    new_particles = get_empty_particle_df()
    stoss_mask = get_stoss_mask(stoss_range, U_x, U_z, X)
    if stoss_mask.any():
        stoss_normal_flux = get_stoss_normal_flux(stoss_range, profile_func, U_x, U_z, X)
        flow_is_inward = get_flow_is_inward(stoss_range, U_x, U_z, X, profile_func)
        if flow_is_inward.any():
            num_stoss_nodes = len(flow_is_inward)
            stoss_inward_flux = stoss_normal_flux[flow_is_inward]
            inward_flux_magnitude = np.apply_along_axis(norm, 1, stoss_inward_flux)

            dx_domain_grid = x_axis[1] - x_axis[0]
            x_grid_incrs = np.repeat(dx_domain_grid, num_stoss_nodes)

            x_stoss = X[stoss_mask]
            f_dz = get_dz_fn(stoss_range, U_x, U_z, X, profile_func)
            dz_stoss = f_dz(x_stoss)

            dx_stoss = get_dx_stoss(stoss_range, U_x, U_z, X)
            ds = np.apply_along_axis(norm, 1, np.vstack([x_grid_incrs, dz_stoss / dx_stoss * x_grid_incrs]).T)
            ds_at_inward_flow_nodes = ds[flow_is_inward]

            inward_flux_cm3 = inward_flux_magnitude * ds_at_inward_flow_nodes * dt * porosity * flume_width_cm
            total_inward_flux_cm3 = inward_flux_cm3.sum()

            num_particles_avg = total_inward_flux_cm3 * surface_MP_conc_ppL/1000
            num_particles = round(np.random.poisson(num_particles_avg))

            # given the number of particles that will enter here, randomly assign them entry nodes, using the inward flow velocity as weighted probabilities for the possible entry nodes
            node_probs = inward_flux_cm3 / total_inward_flux_cm3
            particle_nodes = np.random.choice(len(node_probs), num_particles, p = node_probs)
            particle_start_x = X[stoss_mask][flow_is_inward][particle_nodes]
            particle_start_z = Z[stoss_mask][flow_is_inward][particle_nodes]

            new_particles["X_cm"] = particle_start_x
            new_particles["Z_cm"] = particle_start_z
            new_particles["Deposited"] = np.repeat(False, num_particles)

    return new_particles

def get_new_particles_lee(lee_range, U_x, U_z, celerity_cm_s, dt, surface_MP_conc_ppL, X, Z, porosity = 0.33, flume_width_cm = 30):
    v_nan_right = get_v_nan_right(U_x, U_z)
    lee_mask = in_lee_range(X, lee_range) & v_nan_right

    # when we are excluding margins due to anomalous head behavior, there can arise a case where the non-excluded portion of a lee range near the margin is narrow enough to where it does not contain any lee nodes.
    # this is problematic because then Z[lee_mask] is empty, causing Z[lee_mask].max() to throw an error
    # thus handle this case in the following way:
    # if the lee range contains any lee nodes, compute the lee height
    # otherwise, set the lee height equal to 0
    if lee_mask.any():
        lee_ht = Z[lee_mask].max() - Z[lee_mask].min()
    else:
        lee_ht = 0

    bedform_dx = celerity_cm_s * dt
    turnover_volume_cm3 = lee_ht * bedform_dx * flume_width_cm * porosity

    num_particles_avg = turnover_volume_cm3 * surface_MP_conc_ppL/1000
    num_particles = round(np.random.poisson(num_particles_avg))

    particle_nodes = np.random.choice(len(X[lee_mask]), num_particles)
    particle_start_x = X[lee_mask][particle_nodes]
    particle_start_z = Z[lee_mask][particle_nodes]

    new_particles = pd.DataFrame({"X_cm": particle_start_x,
                                  "Z_cm": particle_start_z,
                                  "Deposited": np.repeat(False, num_particles)})

    return new_particles

def get_lee_particle_mass_mg(lee_mask, celerity_cm_s, dt, porosity, surface_clay_conc_g_L, particles_per_lee_pt):
    """
    Compute the mass of a particle that is entering the bed on the lee side of a bedform, i.e. due to turnover.
    Bulk volume of the porewater that is displaced per timestep due to turnover is computed as the volume of a 3D parallelogram, whose height is the height of the lee face of the bedform, downstream length is the downstream displacement of a bedform per unit time, and width is 1 cm^3. This bulk volume is multiplied by porosity to calculate the corresponding porewater volume.


    Returns
    -------
    None.

    """
    lee_ht = Z[lee_mask].max() - Z[lee_mask].min()
    bedform_dx = celerity_cm_s * dt
    turnover_volume_cm3 = lee_ht * bedform_dx * porosity
    surface_clay_conc_mg_mL = surface_clay_conc_g_L
    turnover_mass_mg = turnover_volume_cm3 * surface_clay_conc_mg_mL
    num_particles = num_lee_pts * particles_per_lee_pt
    particle_mass_mg = turnover_mass_mg / num_particles
    return particle_mass_mg

def area_at_x_for_z_range(x, z_range, x_incr):
    ht = min(profile_func(x), z_range[1]) - z_range[0]
    if ht < 0:
        area  = 0
    else:
        area = ht * x_incr
    return area

def bed_area_for_z_range(z_range, x_range = [0,50], x_incr = 0.1):
    x_coords = np.linspace(x_range[0], x_range[1], int((x_range[1] - x_range[0])/x_incr) + 1)
    x_coords = x_coords[:-1] # drop the last x-coordinate because each x-coordinate should mark the beginning of a segment, and the last element in this array marks the end of a segment
    area = reduce(lambda a,b: a+b, map(lambda x: area_at_x_for_z_range(x, z_range, x_incr), x_coords))
    return area

def get_particle_hts_from_file(filename, x_range = None):
    particles = pd.read_csv(filename)
    z = particles.loc[particles["Deposited"] & (particles["X_cm"] >= x_range[0]) & (particles["X_cm"] <= x_range[1]), "Z_cm"]
    return z

def get_particle_hts_from_df(df, x_range, deposited = None):
    condition = (df["X_cm"] >= x_range[0]) & (df["X_cm"] <= x_range[1])
    if deposited is not None:
        condition = condition & (df["Deposited"] == deposited)
    z = df.loc[condition, "Z_cm"]
    return z

def plot_concentration_vs_depth(particle_dfs, x_range, x_axis, wavelength_cm = 25, height_cm = 2.5, deposited = None):
    particle_ht_arrs = [get_particle_hts_from_df(df, x_range, deposited = deposited) for df in particle_dfs]

    plt.figure();
    count_sets, bins, _ = plt.hist(particle_ht_arrs, bins = np.arange(0, 25.5, 0.5), orientation='horizontal')
    z_ranges = np.vstack([bins[:-1], bins[1:]]).T
    z_range_areas = np.apply_along_axis(bed_area_for_z_range, 1, z_ranges)

    fig = plt.figure()

    # if there is more than one df in particle_dfs, count_sets will be a 2D array (an array of arrays, one per df)
    # if there is only one df in particle_dfs, count_sets will be a 1D array
    # this means we can't use 'for counts in count_sets' for the case of only one df in particle_sets, because then the code will loop over
    # the individual values in the lone array in count_sets (instead of looping over a set of arrays, each of which is a count set)
    # thus we need to handle the one-df case separately
    if len(count_sets.shape) == 2:
        for counts in count_sets:
            counts = counts / z_range_areas * flume_width_cm
            plt.plot(counts, bins[1:], marker = 'o')
    else:
        counts = count_sets
        counts = counts / z_range_areas * flume_width_cm
        plt.plot(counts, bins[1:], marker = 'o')


    ax = plt.gca()
    axis_position_lrwh = [0.125, 0.15, 0.775, 0.8]
    ax.set_position(axis_position_lrwh)
    plt.ylim(14, 25)
    # lgnd = plt.legend(labels = [r"$\lambda_f$ = 0.2", r"$\lambda_f$ = 0.6", 'Max. Scour Depth'], loc = 'lower right')
    plt.ylabel("Z (cm)")
    plt.xlabel(r"Particles Deposited (# of Particles per cm${}^3)$")

    # add in bedform plot for background
    ax2 = ax.twiny()
    ax2.set_position(axis_position_lrwh)
    profile_func_plot = get_bedform_profile_fn(0, wavelength_cm, height_cm, x_axis)
    x_plot = np.linspace(0, 25, 1000)
    bf_plot = ax2.plot(x_plot, profile_func_plot(x_plot), color = 'gray', linestyle='--')
    ax2.get_xaxis().set_visible(False)

    return fig

def plot_step_cel_stn(particles):
    """
    Function to plot the motion of a set of particles, using the coordinate transformation of superimposing a constant celerity on top of the velocity field imposed by a stationary bedform.

    particles: a set of particles as contained in particle_sets in run_model.py
    """

    mobile = np.asarray(~particles["Deposited"].astype(bool))
    particle_U = get_particle_U(particles[["X_cm", "Z_cm"]], U_x, Uz_w_sv, X, Z, masks)
    particle_U[:,0] -= celerity_cm_s
    displacement = get_displacement(particle_U, dt, a_L, a_T)
    q_disp = plt.quiver(particles["X_cm"], particles["Z_cm"], displacement[:,0], displacement[:,1], zorder = 2)
    particles.loc[mobile, ["X_cm", "Z_cm"]] += displacement[mobile]
    s_stn = plt.scatter(particles["X_cm"], particles["Z_cm"], color = 'pink', zorder = 10)

def plot_step_mb(i, particles):
    """
    Function to plot the motion of a set of particles due to a flow field that varies because of bedform motion. (Changes to the bed shape or the bed head field are not plotted, just particle locations at each timestep, along with velocity arrows.)

    i: an int, the number of timestep we are on, starting at 0
    particles: a set of particles as contained in particle_sets in run_model.py
    """
    offset = i * celerity_cm_s * dt # with dt =60, I'm trying to replicate Jon's measured baseline celerity value of 0.1465 cm/min for our flume and sand with no clay

    head_fn = get_head_fn(H_m, wavelength_cm, offset, crest_offset_cm)
    head_top_BC = head_fn(x_axis)

    imposed_head_ax = fig.axes[0]
    imposed_head_ax.lines[0].remove()
    imposed_head_ax.plot(x_axis, head_top_BC, color = 'white')
    imposed_head_ax.set_facecolor('black')

    profile_func = get_bedform_profile_fn(offset, wavelength_cm, height_cm, x_axis)

    masks = get_domain_masks(X, Z, profile_func)
    i_vert, j_vert = get_i_j_vert(masks["head_unknown"])

    update_K(K, masks, Ks_sand)
    impose_head_BCs(head, H_m, wavelength_cm, masks, offset)

    LHS_head_mat_0 = get_trans_head_mat(Ss, dt, K, dx, dz, masks, 'left', 0)
    LHS_inv_0 = sp_inv(LHS_head_mat_0)

    RHS_head_mat_0 = get_trans_head_mat(Ss, dt, K, dx, dz, masks, 'right', 0)
    RHS_vec = get_trans_head_RHS_vec(head, RHS_head_mat_0, masks, Ss, dt, K, dx, dz, 0)

    head[i_vert, j_vert] = LHS_inv_0.dot(RHS_vec)

    LHS_head_mat_1 = get_trans_head_mat(Ss, dt, K, dx, dz, masks, 'left', 1)
    LHS_inv_1 = sp_inv(LHS_head_mat_1)

    RHS_head_mat_1 = get_trans_head_mat(Ss, dt, K, dx, dz,  masks, 'right', 1)
    RHS_vec = get_trans_head_RHS_vec(head, RHS_head_mat_1, masks, Ss, dt, K, dx, dz, 1)

    head[masks["head_unknown"]] = LHS_inv_1.dot(RHS_vec)
    outside_of_bed = np.logical_not(masks["bed"])
    head[outside_of_bed] = np.NaN
    ax = fig.axes[1]
    x = np.linspace(X.min(), X.max(), 1000)
    y = profile_func(x)
    particles_mb = pd.DataFrame(columns = particles.columns)
    particles_mb.loc[0] = np.asarray([37.0, 20.75, False])
    particles_mb.loc[0]["Deposited"] = False
    particles_mb
    dh_dx_mat = get_dh_dx_mat(head, masks, dx)
    dh_dx = get_dh_dx(head, masks, dh_dx_mat, dx)

    dh_dz_mat = get_dh_dz_mat(masks, dz)
    dh_dz = get_dh_dz(head, masks, dh_dz_mat, dz)
    U_x = -(K_x * dh_dx)
    U_z = -(K_z * dh_dz)

    mobile = np.asarray(~particles["Deposited"].astype(bool))
    particle_U = get_particle_U(particles[["X_cm", "Z_cm"]], U_x, Uz_w_sv, X, Z, masks)
    displacement = get_displacement(particle_U, dt, a_L, a_T)
    q_disp = plt.quiver(particles["X_cm"], particles["Z_cm"], displacement[:,0], displacement[:,1], zorder = 2)
    particles.loc[mobile, ["X_cm", "Z_cm"]] += displacement[mobile]
    s_mb = plt.scatter(particles["X_cm"], particles["Z_cm"], color = 'white', zorder = 10)

def get_time_text_str(i, dt):
    min_elapsed = round(i * dt / 60)
    hrs_elapsed = 0
    output = "Time: "
    if min_elapsed >= 60:
        hrs_elapsed = int(min_elapsed / 60)
        remainder_min = min_elapsed % 60
        output += str(hrs_elapsed) + "h " + str(remainder_min).zfill(2) + "m"
    else:
        output += str(min_elapsed) + "m"

    return output

def get_filt_coeff_label(l):
    return r"$\lambda_f = $" + str(l)

def get_sv_label(sv_reduction_factor):
    return r"S.V. / Max $U_z$: 1/" + str(sv_reduction_factor)

def label_from_particle_spec(spec):
    return r"$\lambda_f$: " + str(spec[KEY_FC]) + ", S.V.: " + ('%.2E' % spec[KEY_SV]) + " cm/s"