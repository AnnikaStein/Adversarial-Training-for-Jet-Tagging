# a couple of helpful definitions, e.g. for preparations, training, plotting

import numpy as np

var_names_high_level = ['jet_pt','jet_eta', 'flavor', 'track_2_d0_significance', 'track_3_d0_significance', 'track_2_z0_significance', 'track_3_z0_significance', 'n_tracks_over_d0_threshold', 'jet_prob', 'jet_width_eta', 'jet_width_phi', 'vertex_significance', 'n_secondary_vertices', 'n_secondary_vertex_tracks', 'delta_r_vertex', 'vertex_mass', 'vertex_energy_fraction']
var_text_high_level = [r'Jet $p_t$',r'Jet $\eta$', 'Flavor', r'Track 2 $d_0$ significance', r'Track 3 $d_0$ significance', r'Track 2 $z_0$ significance', r'Track 3 $z_0$ significance', r'Number of tracks over $d_0$ threshold', 'Jet probability', r'Jet width $\eta$', r'Jet width $\phi$', 'Vertex significance', 'Number of secondary vertices', 'Number of tracks at secondary vertex', r'Vertex $\Delta R$', 'Vertex mass', 'Vertex energy fraction']
var_units_high_level = ['GeV','', '', '', '', '', '', '', '', '', '', '', '', '', '', 'GeV', '']
var_digits_high_level = [0,2,2,2,2,2,2,2,4,3,3,2,2,2,2,2,2]
var_ranges_high_level = [[0,300],[None,None],[None,None],[0,2.5],[0,5],[0,5],[0,5],[0,10],[0,0.04],[0,0.4],[0,0.4],[0,5],[0,10],[0,10],[0,7],[0,10],[0,5]]
n_high_level = len(var_names_high_level)

var_names_track_and_vertex_STR = "D0,Z0,PHI,THETA,QOVERP,D0D0,Z0D0,Z0Z0,PHID0,PHIZ0,PHIPHI,THETAD0,THETAZ0,THETAPHI,THETATHETA,QOVERPD0,QOVERPZ0,QOVERPPHI,QOVERPTHETA,QOVERPQOVERP,weight,vertex_mass,vertex_displacement,vertex_delta_eta_jet,vertex_delta_phi_jet,vertex_displacement_significance,vertex_n_tracks,vertex_energy_fraction"
var_names_track_and_vertex = var_names_track_and_vertex_STR.split(",")
var_text_track_and_vertex = [r'$d_0$',r'$z_0$',r'$\phi$',r'$\theta$',r'$Q/P$',r'Cov$(d_0,d_0)$',r'Cov$(z_0,d_0)$',r'Cov$(z_0,z_0)$',r'Cov$(\phi,d_0)$',r'Cov$(\phi,z_0)$',r'Cov$(\phi,\phi)$',r'Cov$(\theta,d_0)$',r'Cov$(\theta,z_0)$',r'Cov$(\theta,\phi)$',r'Cov$(\theta,\theta)$',r'Cov$(Q/P,d_0)$',r'Cov$(Q/P,z_0)$',r'Cov$(Q/P,\phi)$',r'Cov$(Q/P,\theta)$',r'Cov$(Q/P,Q/P)$','weight','vertex mass','vertex displacement',r'vertex-jet $\Delta \eta$',r'vertex-jet $\Delta \phi$','vertex displacement significance','number of tracks at vertex','vertex energy fraction']
var_units_track_and_vertex = ['cm','cm','','',r'GeV${}^{-1}$',r'cm${}^2$',r'cm${}^2$',r'cm${}^2$','cm','cm','','cm','cm','','',r'cm GeV${}^{-1}$',r'cm GeV${}^{-1}$',r'GeV${}^{-1}$',r'GeV${}^{-1}$',r'GeV${}^{-1}$ * GeV${}^{-1}$','','GeV','cm','','','','','']
var_digits_track_and_vertex = [3,3,3,3,2,5,5,5,5,5,5,5,5,5,5,5,5,5,5,5,2,2,2,2,2,2,2,2]
var_ranges_track_and_vertex = [[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[-0.2,0.2],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None],[None,None]]

a4_indices_list = [k for k in range(3,11)]
a5_indices_list = [k for k in range(11,17)]

all_low_level_indices = (np.array([[(i * 33 + k) for k in range(33)] for i in range(28)])).flatten()

non_target_indices = [0,1]
non_target_after_target = [k for k in range(3,941)]
non_target_indices.extend(non_target_after_target)

n_input_features = len(non_target_indices)

all_factor_epsilons = []
factor_epsilons_hl = [2.5,2,#ok
                      0,#ok
                      4,4,4,4,#ok
                      3,2,#ok
                      3,3,#ok
                      3,5,1,2,1,3]#ok
factor_epsilons_track_and_vertex = [3,3,3,3,3,
                                    2,
                                    2,2,
                                    2,2,2,
                                    2,2,2,2,
                                    2,2,2,2,2,
                                    2,
                                    1,3,2,2,4,1,3] * 33
all_factor_epsilons.extend(factor_epsilons_hl)
all_factor_epsilons.extend(factor_epsilons_track_and_vertex)

def track_vertex_index_to_name(index):
    
    var, track = divmod(index, 33)
    # note: use 0-indexing here
    name = 'track_'+ str(track) + '_' + var_names_track_and_vertex[var]
    
    return name

def track_vertex_index_to_text(index):
    
    var, track = divmod(index, 33)
    # note: instead of using 0-indexing for the text, starting at 1 will be easier to understand (index 1 = first track etc.)
    name = 'Track '+ str(track+1) + ' ' + var_text_track_and_vertex[var]
    
    return name

def track_vertex_index_to_unit(index):
    
    var, track = divmod(index, 33)
    unit = var_units_track_and_vertex[var]
    
    return unit

def track_vertex_index_to_digit(index):
    
    var, track = divmod(index, 33)
    digit = var_digits_track_and_vertex[var]
    
    return digit

def track_vertex_index_to_range(index):
    
    var, track = divmod(index, 33)
    ranges = var_ranges_track_and_vertex[var]
    
    return ranges

def full_index_to_name(index):
    
    if index >= n_high_level:
        name = track_vertex_index_to_name(index - n_high_level)
    else:
        name = var_names_high_level[index]
        
    return name 

def full_index_to_text(index):
    
    if index >= n_high_level:
        name = track_vertex_index_to_text(index - n_high_level)
    else:
        name = var_text_high_level[index]
        
    return name 

def full_index_to_unit(index):
    
    if index >= n_high_level:
        unit = track_vertex_index_to_unit(index - n_high_level)
    else:
        unit = var_units_high_level[index]
        
    return unit

def full_index_to_digit(index):

    if index >= n_high_level:
        digits = track_vertex_index_to_digit(index - n_high_level)
    else:
        digits = var_digits_high_level[index]
        
    return digits

    
def full_index_to_range(index):
    
    if index >= n_high_level:
        ranges = track_vertex_index_to_range(index - n_high_level)
    else:
        ranges = var_ranges_high_level[index]
        
    return ranges

def get_track_vertex_indices_all_tracks_one_variable(index):
    return [index * 33 + k for k in range(33)]

def get_track_vertex_index_from_full_index(index):
    return (index - n_high_level)
    
def get_full_index_from_track_vertex_index(index):
    return (index + n_high_level)

def get_var_and_track_from_track_vertex_index(index):
    var, track = divmod(index, 33)
    return var, track

def get_non_target_index_from_full_index(index):
    if index < 2:
        return index
    elif index == 2:
        raise ValueError('Can not return non-target index for the actual target index!')
    else:
        return index-1
    
def get_full_index_from_non_target_index(index):
    if index < 2:
        return index
    else:
        return index+1
    
def input_indices_wanted(n_highlevel = n_high_level-1, n_tracks=6, n_track_vertex=28):
    relevant_columns_overall = (np.array([[(i * 33 + k) for k in range(n_tracks)] for i in range(n_track_vertex)])).flatten()
    relevant_columns_overall = np.concatenate([np.arange(n_highlevel), relevant_columns_overall+n_highlevel])
    return relevant_columns_overall

def get_wanted_full_indices(filtered_indices):
    return np.array([get_full_index_from_non_target_index(k) for k in filtered_indices])

# note: 2 is trivial = target column
highlevel_integer_indices = [2, 7, 12, 13]
# there are 28 vars, 27 is the last index, and 26 points to the index just before that one (vertex_n_tracks)
lowlevel_integer_indices = [get_full_index_from_track_vertex_index(k) for k in get_track_vertex_indices_all_tracks_one_variable(26)]
integer_indices = highlevel_integer_indices + lowlevel_integer_indices
