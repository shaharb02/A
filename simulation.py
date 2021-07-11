#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:44:20 2019

This project implements a particle-tracking model of clay deposition in a sandy stream bed.

This file actually runs the model, using functions defined in clay_dep_cont.py.

@author: yoni
"""
import os
import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
if not script_dir in sys.path:
    sys.path.append(script_dir) # append the directory that houses this script to the path

from numpy.linalg import inv
from scipy.sparse.linalg import spsolve
from matplotlib.animation import FuncAnimation
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import pandas as pd
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle

from clay_dep_cont import *

prop_cycle = plt.rcParams['axes.prop_cycle']
colors = prop_cycle.by_key()['color']

red = to_rgba('r', 1)
green = to_rgba('g', 1)
magenta = to_rgba('m', 1)
cyan = to_rgba('c', 1)
tab_blue = to_rgba('tab:blue', 1)
tab_orange = to_rgba('tab:orange', 1)
tab_purple = to_rgba('tab:purple', 1)
tab_pink = to_rgba('tab:pink', 1)
tab_gray = to_rgba('tab:gray', 1)
tab_olive = to_rgba('tab:olive', 1)

particle_colors = [red, green, magenta, cyan, tab_blue, tab_orange, tab_purple, tab_pink, tab_gray, tab_olive]
markers = ['o', 's', 'd', 'X', 'v', '^', '<', '>', '1', '2']

class Simulation():

    KEY_FC = "filtration_coefficient"
    KEY_SV = "settling_velocity"

    def __init__(self):
        pass

    def restore():
        pass

    def from_file():
        pass

    @staticmethod
    def from_object(config):
        s = Simulation()

        # number of segments to divide domain along x and z axes, respectively
        s.J = config["J"]
        s.I = config["I"]

        s.x_min = config["x_min"]
        s.x_max = config["x_max"]

        s.z_min = config["z_min"]
        s.z_max = config["z_max"]

        s.dx = (s.x_max - s.x_min) / s.J
        s.dz = (s.z_max - s.z_min) / s.I

        x_axis = np.linspace(s.x_min, s.x_max, s.I+1)
        z_axis = np.linspace(s.z_min, s.z_max, s.J+1)
        X, Z = np.meshgrid(x_axis, z_axis)
        s.X = X
        s.Z = Z

        s.displacement_per_timestep_cm = config["displacement_per_timestep_cm"]

        s.celerity_cm_min = config["celerity_cm_min"]
        s.celerity_cm_s = s.celerity_cm_min / 60

        # the length of time we want to simulate, in minutes
        s.sim_duration_min = config["sim_duration_min"]

        # size of the timestep, in seconds
        s.dt = s.displacement_per_timestep_cm / s.celerity_cm_s

        # number of frames to include in the animation that will be generated
        s.anim_num_frames = int(round((s.sim_duration_min * 60 / s.dt)))

        # the ratio of how much simulation time should pass per unit of real time
        # for example, if I set this value to 180, it means that I want 180 minutes of simulated time to pass in 1 minute of real time
        # used in generating the animation of a simulation
        s.sim_time_ratio = config["sim_time_ratio"]

        # the how long, in milliseconds, one frame of the animation should last for this simulation
        s.anim_frame_interval_ms = int(round((s.dt / s.sim_time_ratio) * 1000))

        # Specific storage
        # Since I am treating the sand bed as an 'unconfined aquifer', I just use the recorded porosity for our sand, which is 0.33
        s.Ss = config["porosity"]

        # number of levels for making the contour plot of the bed
        s.contour_num_levels = 25

        # since head behaves anomalously near side boundary, leave a margin at either side boundary when doing particle tracking calculations
        s.head_margin_cm = config["head_margin_cm"]

        # saturated hydraulic conductivity of the bed
        # units: cm s-1
        s.Ks_sand = config["Ks_sand"]
        s.K_x = s.Ks_sand
        s.K_z = s.Ks_sand

        s.water_depth_cm = config["water_depth_cm"]

        # bedform shape properties
        s.wavelength_cm = config["wavelength_cm"]
        s.bedform_height_cm = config["bedform_height_cm"]
        s.crest_offset_cm = config["crest_offset_cm"]

        s.porosity = config["porosity"]
        s.flume_width_cm = config["flume_width_cm"]

        s.particle_specs = config["particle_specs"]

        # dispersivity
        # a_L = 0.01 # 1 m
        s.a_L = 10
        s.a_T = s.a_L * 0.1

        # get stream water flow velocity, in m/s
        V_from_cel = get_V_fn(s.water_depth_cm / 100)
        s.V = V_from_cel(s.celerity_cm_min)

        s.H_m = get_ElliottBrooks_Hm(s.V, s.bedform_height_cm, s.water_depth_cm)

        # this will actually need to be an array containing particle concentrations per liter
        s.surface_MP_conc_ppL = 500

        s.particles_released = np.zeros(len(s.particle_specs))

        s.particle_sets = []

        for i in range(len(s.particle_specs)):
            spec = s.particle_specs[i]
            particles = get_empty_particle_df()
            s.particle_sets.append(particles)

        return s

    def get_Ux_and_Uz(self):
        dh_dx_mat = get_dh_dx_mat(self.head, self.masks, self.dx)
        dh_dx = get_dh_dx(self.head, self.masks, dh_dx_mat, self.dx)

        dh_dz_mat = get_dh_dz_mat(self.masks, self.dz)
        dh_dz = get_dh_dz(self.head, self.masks, dh_dz_mat, self.dz)

        self.U_x = -(self.K_x * dh_dx) / self.porosity
        self.U_z = -(self.K_z * dh_dz) / self.porosity

    def compute_initial_conditions(self):
        self.t = 0
        self.offset_cm = 0
        self.profile_func = get_bedform_profile_fn(self.offset_cm, self.wavelength_cm, self.bedform_height_cm, self.crest_offset_cm, self.X[0])
        self.masks = get_domain_masks(self.X, self.Z, self.profile_func)

        self.K = np.tile(np.NaN, self.X.shape)
        self.K[self.masks["bed"]] = self.Ks_sand

        # solve steady-state GW flow equation for initial head in the bed
        self.head = np.tile(np.NaN, self.X.shape)
        self.head[self.masks["bed"]] = 0
        head_fn = get_head_fn(self.H_m, self.wavelength_cm, self.offset_cm, self.crest_offset_cm)
        impose_head_BCs(self.head, self.H_m, self.wavelength_cm, self.masks, self.crest_offset_cm, head_fn, self.X, self.Z, self.offset_cm, side_BC = "ExpDecay")
        A = get_initial_head_mat(self.K, self.dx, self.dz, self.masks)
        b = get_head_RHS(self.head, self.K, self.dx, self.dz, self.masks)
        self.head[self.masks["head_unknown"]] = spsolve(A, b)

        # compute flow paths using Darcy's law
        self.get_Ux_and_Uz()

    def plot_initial_conditions(self):
        fig = plt.figure()
        plt.contourf(self.X, self.Z, self.head)
        plt.colorbar()
        plt.title("Head in the Bed, t0, Before Optimization")
        #plt.show()
        return fig



        # currently we use homogeneous K
        # for heterogeneous K do the following:
        # K_z = harm_mean([K, np.roll(K, -1, 0), np.roll(K, 1, 0)])
        # K_z[0,:] = harm_mean([K, np.roll(K, -1, 0)])[0] # for the bottom row of K, take the "upward" harmonic mean consisting only of that row and the row above, since we don't have a row below to incorporate.
        # K_x = harm_mean([K, np.roll(K, -1, 1)])

    def update_domain_shape(self):
        self.profile_func = get_bedform_profile_fn(self.offset_cm, self.wavelength_cm, self.bedform_height_cm, self.crest_offset_cm, self.X[0])
        self.masks = get_domain_masks(self.X, self.Z, self.profile_func)

    def update_K(self):
        """
        Function to update K values in the bed in light of erosion/deposition of sand along the top boundary.
        This function updates K in-place (i.e. modifies the array that is passed in, instead of making a copy of it and returning a new array).

        NOTE: for right now this function assumes that no significant amount of clay has deposited at any new nodes, i.e. that a new node starts with the default K_s value for the sand we used. (The alternative would be if, for example, the timestep used in the model was very large such that realistically a new node would have enough time to accumulate enough clay to reduce K_s by the "time" of initializing that node's K value.)
        """
        # initialize K in nodes where sand has been newly deposited (i.e. at nodes that are newly a part of the bed)
        self.K[self.masks["bed"] & np.isnan(self.K)] = self.Ks_sand

        # set K to NaN where the bed has been eroded away (i.e. at nodes that are no longer part of the bed)
        self.K[self.masks["non_bed"] & ~np.isnan(self.K)] = np.NaN

    def solve_transient_GW_flow_eq(self):
        head_fn = get_head_fn(self.H_m, self.wavelength_cm, self.offset_cm, self.crest_offset_cm)
        impose_head_BCs(self.head, self.H_m, self.wavelength_cm, self.masks, self.crest_offset_cm, head_fn, self.X, self.Z, self.offset_cm, side_BC = "ExpDecay")

        LHS_head_mat_0 = get_trans_head_mat(self.Ss, self.dt, self.K, self.dx, self.dz, self.masks, 'left', 0)
        RHS_head_mat_0 = get_trans_head_mat(self.Ss, self.dt, self.K, self.dx, self.dz, self.masks, 'right', 0)
        RHS_vec = get_trans_head_RHS_vec(self.head, RHS_head_mat_0, self.masks, self.Ss, self.dt, self.K, self.dx, self.dz, 0)

        i_vert, j_vert = get_i_j_vert(self.masks["head_unknown"])
        self.head[i_vert, j_vert] = spsolve(LHS_head_mat_0, RHS_vec)

        LHS_head_mat_1 = get_trans_head_mat(self.Ss, self.dt, self.K, self.dx, self.dz, self.masks, 'left', 1)
        RHS_head_mat_1 = get_trans_head_mat(self.Ss, self.dt, self.K, self.dx, self.dz, self.masks, 'right', 1)
        RHS_vec = get_trans_head_RHS_vec(self.head, RHS_head_mat_1, self.masks, self.Ss, self.dt, self.K, self.dx, self.dz, 1)
        self.head[self.masks["head_unknown"]] = spsolve(LHS_head_mat_1, RHS_vec)

        outside_of_bed = np.logical_not(self.masks["bed"])
        self.head[outside_of_bed] = np.NaN

    def update_particles(self):
        self.get_Ux_and_Uz()

        x_axis = self.X[0]
        bedform_stoss_ranges = get_bedform_stoss_ranges(self.offset_cm, self.wavelength_cm, x_axis, self.crest_offset_cm)
        bedform_lee_ranges = get_bedform_lee_ranges(self.offset_cm, self.wavelength_cm, x_axis, self.crest_offset_cm)

        for j in range(len(self.particle_sets)):
            spec = self.particle_specs[j]
            particles = self.particle_sets[j]
            settling_velocity = spec[Simulation.KEY_SV]
            l = spec[Simulation.KEY_FC]
            Uz_w_sv = self.U_z.copy() - settling_velocity

            # get particles that have newly entered the bed due to HEF or turnover
            new_particles_stoss = pd.concat([get_new_particles_stoss(stoss_range, self.profile_func, self.U_x, self.U_z, self.surface_MP_conc_ppL, self.X, x_axis, self.Z, self.dt) for stoss_range in bedform_stoss_ranges], ignore_index = True)
            new_particles_lee = pd.concat([get_new_particles_lee(lee_range, self.U_x, self.U_z, self.celerity_cm_s, self.dt, self.surface_MP_conc_ppL, self.X, self.Z, self.porosity, self.flume_width_cm) for lee_range in bedform_lee_ranges], ignore_index = True)
            num_new_particles = new_particles_stoss.shape[0] + new_particles_lee.shape[0]
            self.particles_released[j] += num_new_particles
            particles = pd.concat([particles, new_particles_stoss, new_particles_lee], ignore_index = True)

            # if there are any particles in the domain...
            if len(particles.index) > 0:

                # propagate particles due to porewater flow velocities
                mobile = np.asarray(~particles["Deposited"].astype(bool) & ~np.isnan(particles["X_cm"]) & ~np.isnan(particles["Z_cm"]))
                if mobile.any():
                    particle_U = get_particle_U(particles.loc[mobile, ["X_cm", "Z_cm"]], self.U_x, Uz_w_sv, self.X, self.Z, self.masks)
                    displacement = get_displacement(particle_U, self.dt, self.a_L, self.a_T)
                    particles.loc[mobile, ["X_cm", "Z_cm"]] += displacement
                    particles_below_base = particles["Z_cm"] < 0
                    particles.loc[particles_below_base, "Z_cm"] = 0
                    particles.loc[mobile, "Deposited"] = update_dep(particles.loc[mobile, "Deposited"], displacement, l)

                # handle particles that have exited the bed due to either turnover or pumping
                if len(particles) > 0:
                    exited_bed = (particles["Z_cm"] > self.profile_func(particles["X_cm"]))
                    in_bed = np.logical_not(exited_bed)
                    particles = particles.loc[in_bed]
                    self.particle_sets[j] = particles

    def update(self, t):
        print("i = " + str(t) + ", num_frames = " + str(self.anim_num_frames) + ", " + str(round(100*t/self.anim_num_frames, 2)) + "% Complete")
        self.offset_cm = t * self.celerity_cm_s * self.dt
        self.update_domain_shape()
        self.update_K()

        self.solve_transient_GW_flow_eq()
        self.update_particles()
        # if should_plot:
        #     update_plot()

#     def update_head_BCs():
#         head_fn = get_head_fn(H_m, wavelength_cm, offset, crest_offset_cm)
#         x_axis = self.X[0]
#         head_top_BC = head_fn(self.X[0])
#         impose_head_BCs(self.head, self.H_m, self.wavelength_cm, masks, self.crest_offset_cm, head_fn, offset, side_BC = "ExpDecay")

#     def get_optimized_head():
#         print("Optimizing head at bed surface...")
#         runfile('C:/Users/user/Google Drive/clay_dep_cont/head_optimization_for_trough_slice.py', wdir='C:/Users/user/Google Drive/clay_dep_cont', current_namespace=True)
#         res = minimize(fn_to_minimize, head_top_BC, args=(A, wavelength_cm), options = {"disp": True})
#         optimized_head = res.x



#     def get_Ux_and_Uz():
#         dh_dx_mat = get_dh_dx_mat(head, masks, dx)
#         dh_dx = get_dh_dx(head, masks, dh_dx_mat, dx)

#         dh_dz_mat = get_dh_dz_mat(masks, dz)
#         dh_dz = get_dh_dz(head, masks, dh_dz_mat, dz)

#         # currently we use homogeneous K
#         # for heterogeneous K do the following:
#         # # K_z = harm_mean([K, np.roll(K, -1, 0), np.roll(K, 1, 0)])
#         # # K_z[0,:] = harm_mean([K, np.roll(K, -1, 0)])[0] # for the bottom row of K, take the "upward" harmonic mean consisting only of that row and the row above, since we don't have a row below to incorporate.
#         # # K_x = harm_mean([K, np.roll(K, -1, 1)])
#         K_x = K
#         K_z = K

#         U_x = -(K_x * dh_dx)
#         U_z = -(K_z * dh_dz)



#     def update_plot():


#         # update head top boundary condition at the top of the plot
#         imposed_head_ax = fig.axes[0]
#         imposed_head_ax.lines[0].remove()
#         imposed_head_ax.plot(x_axis, head_top_BC, color = 'white')
#         imposed_head_ax.set_facecolor('black')

#         # update head contour plot
#         ax = plt.gca()
#         plt.cla()
#         ax.set_facecolor('black')

#         x = np.linspace(X.min(), X.max(), 1000)
#         y = profile_func(x)
#         profile_line = ax.plot(x, y, color = 'white', label = 'Bed Profile')

#         plt.contourf(X, Z, head, 15)

#         plt.xlim(x_min, x_max)
#         plt.ylim(z_min, z_max)
#         plt.xlabel("Distance Downstream (cm)")
#         plt.ylabel("Height (cm)")

#         # plot particles

#         # determine visual representation of each particle based on whether or not it has deposited
#         color = np.asarray([particle_colors[j]] * particles.shape[0])
#         dep_color = np.asarray(particle_colors[j]).copy()
#         dep_color[:3] += (np.ones(3) - dep_color[:3]) * 0.75
#         edgecolors = color.copy()
#         edgecolors[:,:3] *= 0.5 # make borders darker
#         if particles["Deposited"].any():
#             # color[particles["Deposited"], 3] = 0.25
#             color[particles["Deposited"].astype(bool)] = dep_color
#             edgecolors[particles["Deposited"].astype(bool), :3] = 0

#         # plot the particles and keep the handle for the artist
#         particle_artist = plt.scatter(particles["X_cm"], particles["Z_cm"], marker = markers[j], c = color, edgecolors = edgecolors, label = label_from_particle_spec(spec))

#         ax1_bbox = fig.axes[1].get_position()
#         arrow_x_loc = (ax1_bbox.xmax + ax1_bbox.xmin) / 2
#         flow_text = plt.text(arrow_x_loc, 0.9, 'Flow Direction', ha = 'center', fontsize = 14, bbox = dict(boxstyle='rarrow,pad=0.2', fc = 'xkcd:lightblue', ec = 'black', lw = 1), transform = plt.gcf().transFigure)




# # ## EXECUTION OF MODEL STARTS HERE ###

# # should_plot = False



# anim = FuncAnimation(fig, update_w_plot, frames = num_frames, interval = anim_frame_interval_ms, repeat = False)
# anim = FuncAnimation(fig, update, frames = num_frames, interval = 100)
# plt.show()




#
#gif_name = "C:/Users/user/Google Drive/clay_dep_cont/moving_bed_w_head_field.gif"
#gif_name = "C:/Users/user/Google Drive/clay_dep_cont/moving_bed_w_flow_paths.gif"
# gif_name = "C:/Users/user/Google Drive/clay_dep_cont/particles_moving_bed_filt_0_2_cel_0_45_cm_min_svrf_" + str(sv_reduction_factor) + ".gif"
# gif_name = "C:/Users/user/Google Drive/AGU 2020/presentation/images/particles_moving_bed_filt_0_6_cel_0_15_cm_min.gif"
# gif_name = "C:/Users/user/Google Drive/AGU 2020/presentation/images/test.gif"
# gif_name = "C:/Users/user/Google Drive/Simulating Layer Accumulation/Simulation Results/Pumping Dominated/SV_0_cm_s_Lf_0_180_min.gif"
#gif_name = "C:/Users/user/Google Drive/Simulating Layer Accumulation/Simulation Results/Pumping Dominated/SV_0_cm_s_Lf_0_6_cel_0_45_cm_min_180_min.gif"
# gif_name = "C:/Users/user/Google Drive/Simulating Layer Accumulation/Simulation Results/SV_0_cm_s_Lf_0_6_cel_0_15_cm_min_180_min_K_1_2_cm_s.gif"
# gif_name = "C:/Users/user/Google Drive/Simulating Layer Accumulation/images/Supporting Information/SV_0_cm_s_Lf_0_5_cel_0_15_cm_min_180_min_K_0_12_cm_s.gif"

# gif_name = "C:/Users/user/Google Drive/Simulating Layer Accumulation/images/test_cel_0_01_cm_min_dt_60_s_I_150_J_150_w_particles.gif"
# anim.save(gif_name)
