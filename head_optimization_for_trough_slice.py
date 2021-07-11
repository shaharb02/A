# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:19:28 2021

@author: user
"""

from scipy.optimize import minimize

trough_slice = (Z == Z[(Z <= 20)].max())

head_fn = get_head_fn(H_m, wavelength_cm, offset, crest_offset_cm)

head_top_BC = head_fn(x_axis[x_axis < wavelength_cm])

opt_b, opt_c, opt_d, opt_e = get_head_mat_coeffs(K, dx, dz)

head_trough_slice_orig = head[trough_slice].copy()

def plot_optimization_result(orig_head_top_BC, new_head_top_BC, head_trough_slice_orig, head_trough_slice_new):
    opt_plot = plt.figure();
    num_values = orig_head_top_BC.shape[0]
    ind = np.arange(num_values)
    plt.scatter(ind, orig_head_top_BC, label = 'E&B Head')
    plt.scatter(ind, new_head_top_BC, label = 'New Head Top BC')
    plt.scatter(ind, head_trough_slice_orig[:num_values], label = 'Head Trough Slice Original')
    plt.scatter(ind, head_trough_slice_new[:num_values], label = 'Head Trough Slice New')
    plt.legend(loc = 'best')
    return opt_plot

def get_head_RHS_for_opt(head, K, dx, dz, masks, b, c, d, e, offset = 0):
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

def fn_to_minimize(head_top_BC, A, wavelength_cm):

    ### move to a separate function for getting head boundary conditions

    side_BC_decay_rate = 2 * np.pi / wavelength_cm

    head_top_BC_full = np.resize(head_top_BC, head.shape[1])

    bed_profile_nodes = np.asarray(mask_to_node_list(masks["bed_profile"]))
    i = bed_profile_nodes[:,0]
    j = bed_profile_nodes[:,1]
    head[i, j] = head_top_BC_full[j]

    is_left_BC_node = masks["bed"] & masks["left_boundary"]
    z_left_BC = Z[is_left_BC_node]
    depth_left_BC = z_left_BC.max() - z_left_BC[::-1]
    left_atten = np.exp(-side_BC_decay_rate * depth_left_BC)
    head_left_BC = head[is_left_BC_node][-1] * left_atten # we can do this because head[is_left_BC_node][-1] lies along the top profile of the bed, and the BC has already been set for those nodes
    head[is_left_BC_node] = head_left_BC[::-1] # reverse ordering is necessary because head_left_BC is computed from the top down, but head[is_left_BC_node] is filled in from the bottom up

    is_right_BC_node = masks["bed"] & masks["right_boundary"]
    z_right_BC = Z[is_right_BC_node]
    depth_right_BC = z_right_BC.max() - z_right_BC[::-1]
    right_atten = np.exp(-side_BC_decay_rate * depth_right_BC)
    head_right_BC = head[is_right_BC_node][-1] * right_atten # we can do this because head[is_right_BC_node][-1] lies along the top profile of the bed, and the BC has already been set for those nodes
    head[is_right_BC_node] = head_right_BC[::-1]

    # A = get_initial_head_mat(K, dx, dz, masks)
    b = get_head_RHS_for_opt(head, K, dx, dz, masks, opt_b, opt_c, opt_d, opt_e)
    # head[masks["head_unknown"]] = sp_inv(A).dot(b)
    head[masks["head_unknown"]] = spsolve(A, b)

    a = 0.3
    trough_slice_error = np.linalg.norm(head[trough_slice] - head_fn(x_axis))
    dist_between_pts = np.linalg.norm(np.roll(head_top_BC, 1) - head_top_BC)
    score = a * trough_slice_error + (1-a) * dist_between_pts
    return score