#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# Authors: Daniel van de Velden (d.vandevelden@yahoo.de)
#
# License: BSD (3-clause)

from jumeg.jumeg_utils import loadingBar
import mne
import numpy as np
from scipy.optimize import leastsq
from sklearn.neighbors import BallTree
from mne.transforms import (rotation, rotation3d, scaling,
                            translation, apply_trans)
from mne.source_space import _get_lut, _get_lut_id, _get_mgz_header
import nibabel as nib
from scipy import linalg
from scipy.spatial.distance import cdist
import time
import os.path
from scipy.interpolate import griddata
from mne.source_estimate import _write_stc
from nilearn import plotting
from nilearn.image import index_img
from nibabel.affines import apply_affine
import matplotlib.pyplot as plt
from mne.source_estimate import VolSourceEstimate
from matplotlib import  cm
from matplotlib.ticker import LinearLocator



def _point_cloud_error_balltree(subj_p, temp_tree):
    """Find the distance from each source point to its closest target point.
    Uses sklearn.neighbors.BallTree for greater efficiency"""
    dist, _ = temp_tree.query(subj_p)
    err = dist.ravel()
    return err


def _point_cloud_error(src_pts, tgt_pts):
    """Find the distance from each source point to its closest target point.
    Parameters.
    """
    Y = cdist(src_pts, tgt_pts, 'euclidean')
    dist = Y.min(axis=1)
    return dist


def _trans_from_est(params):
    """Convert transformation parameters into a transformation matrix. """
    i = 0
    trans = []
    x, y, z = params[:3]
    trans.insert(0, translation(x, y, z)) 
    i += 3
    x, y, z = params[i:i + 3]
    trans.append(rotation(x, y, z))
    i += 3
    x, y, z = params[i:i + 3]
    trans.append(scaling(x, y, z))
    i += 3
    trans = reduce(np.dot, trans)
    return trans
  

def auto_match_labels(fname_subj_src, fname_temp_src, volume_labels,
                      template_spacing, subject_dir, e_func,
                      save_trans=False):
    """ Matches a subjects volume source space labelwise to another volume 
        source space
    
    Parameters
    ----------
    fname_subj_src : String
          Filename of the first volume source space
    fname_temp_src : list of Labels
          Filename of the second volume source space to match on
    volume_labels : list of volume Labels
          List of the volume labels of intrest
    template_spacing : float
          The grid distances of the second volume source space in mm
    subject_dir : String
          subjects directory
    e_func : String | None
          Error function, either 'balltree' or 'euclidean'
    save : Boolean
          True to save the transformation matrix for each label as a dictionary.
          False is default
  
    Returns
    -------
    label_trans_dic : dict
          Dictionary of all the labels transformation matrizes
    label_trans_dic_err : dict
          Dictionary of all the labels transformation matrizes distance errors (mm)
    label_trans_dic_mean_dist : dict
          Dictionary of all the labels transformation matrizes mean distances (mm)
    label_trans_dic_max_dist : dict
          Dictionary of all the labels transformation matrizes max distance (mm)
    label_trans_dic_var_dist : dict
          Dictionary of all the labels transformation matrizes distance error variance (mm)
    """
    print '\n    ######################################'
    print '####                  START                ####'
    print '####          LW Matching Function         ####'
    print '    ######################################'

    print '\n#### Attempting to read the volume source spaces..'
    subj_src = mne.read_source_spaces(fname_subj_src)
    subject_from = subj_src[0]['subject_his_id']
    x, y, z = subj_src[0]['rr'].T
    subj_p = np.c_[x, y, z]
    subject = subj_src[0]['subject_his_id']
    fname_labels_subj = fname_subj_src[:-4] + '_vertno_labelwise.npy'
    label_list_subject = np.load(fname_labels_subj).item()
    fname_s_aseg = subject_dir + subject + '/mri/aseg.mgz'
    
    mgz = nib.load(fname_s_aseg)
    mgz_data = mgz.get_data()
    lut = _get_lut()
    vox2rastkr_trans = _get_mgz_header(fname_s_aseg)['vox2ras_tkr']
    vox2rastkr_trans[:3] /= 1000.
    inv_vox2rastkr_trans = linalg.inv(vox2rastkr_trans)
    all_volume_labels = []
    vol_lab = mne.get_volume_labels_from_aseg(fname_s_aseg)
    for lab in vol_lab: all_volume_labels.append( lab.encode() )
    
    print """\n#### Attempting to check for vertice duplicates in labels
       due to spatial aliasing in volume source creation.."""
    del_count = 0
    for p, label in enumerate(all_volume_labels):
      loadingBar(p+1, len(all_volume_labels), task_part=None)
      lab_arr = label_list_subject[label]
      lab_id = _get_lut_id(lut, label, True)[0]
      del_ver_idx_list = [] 
      for arr_id, i in enumerate(lab_arr, 0):
        lab_vert_coord = subj_src[0]['rr'][i]
        lab_vert_mgz_idx = apply_trans(inv_vox2rastkr_trans, lab_vert_coord)
        orig_idx = mgz_data[int( round(lab_vert_mgz_idx[0]) ),
                            int( round(lab_vert_mgz_idx[1]) ),
                            int( round(lab_vert_mgz_idx[2]) ) ]
        if orig_idx != lab_id:
          del_ver_idx_list.append(arr_id)
          del_count += 1     
      del_ver_idx = np.asarray(del_ver_idx_list)
      label_list_subject[label] = np.delete(label_list_subject[label], del_ver_idx)
    print '    ---> Deleted', del_count,'vertice duplicates.'
    
    print '\n#### Attempting to read the volume source spaces to match..'
    temp_src = mne.read_source_spaces(fname_temp_src)
    subject_to = temp_src[0]['subject_his_id']
    x1, y1, z1 = temp_src[0]['rr'].T
    temp_p = np.c_[x1, y1, z1]
    template = temp_src[0]['subject_his_id']
    fname_labels_temp = fname_temp_src[:-4] + '_vertno_labelwise.npy'
    label_list_template = np.load(fname_labels_temp).item()
    del temp_src
    
    vert_sum = []
    vert_sum_temp = []
    for i in volume_labels:
      vert_sum.append(label_list_subject[i].shape[0])
      vert_sum_temp.append(label_list_template[i].shape[0])
      for j in volume_labels:
        if i != j:
          h = np.intersect1d(label_list_subject[i], label_list_subject[j])
          if h.shape[0] > 0:
            print 'In Label:', i, ' are vertices from Label:', j,'(', h.shape[0],')'
            break
    del subj_src
    print '    Number of subject vertices:', np.sum(np.asarray(vert_sum))
    print '    Number of template vertices:', np.sum(np.asarray(vert_sum_temp))
    
    if e_func == 'balltree':
      err_function = 'BallTree Error Function'
      errfunc = _point_cloud_error_balltree
    if e_func == 'euclidean':
      err_function = 'Euclidean Error Function'
      errfunc = _point_cloud_error
    if e_func == None:
      print '\nPlease provide your desired Error Function\n'
    
    print """\n#### Attempting to match %d volume source spaces labels of
       Subject: '%s' to Template: '%s' with
       %s...""" %(len(volume_labels), subject, template, err_function)
    
    label_trans_dic = {}
    label_trans_dic_err = {}
    label_trans_dic_var_dist = {}
    label_trans_dic_mean_dist = {}
    label_trans_dic_max_dist = {}
    start_time = time.time()
    for p, label in enumerate(volume_labels):
      loadingBar(count=p, total=len(volume_labels),
                 task_part='%s' %label)
      # select coords for label and check if they exceed the label size limit
      s_pts = subj_p[label_list_subject[label]]
      t_pts = temp_p[label_list_template[label]]
      
      if s_pts.shape[0] < 6:
        while s_pts.shape[0] < 6:
          s_pts = np.concatenate((s_pts, s_pts))
      
      if t_pts.shape[0] == 0:
        # Append the Dictionaries with the zeros since there is no label to match the points
        trans = _trans_from_est(np.zeros([9, 1]))
        trans[0,0], trans[1,1], trans[2,2] = 1., 1., 1.
        label_trans_dic.update({label:trans})
        label_trans_dic_mean_dist.update({label:np.min(0)})
        label_trans_dic_max_dist.update({label:np.min(0)})
        label_trans_dic_var_dist.update({label:np.min(0)})
        label_trans_dic_err.update({label:0})
      else:
        if e_func == 'balltree':
          temp_tree = BallTree(t_pts)
        if e_func == 'euclidean':
          continue
        # Get the x-,y-,z- min and max Limits to create the span for each axis
        s_x, s_y, s_z = s_pts.T
        s_x_diff = np.max(s_x) - np.min(s_x)
        s_y_diff = np.max(s_y) - np.min(s_y)
        s_z_diff = np.max(s_z) - np.min(s_z)
        t_x, t_y, t_z = t_pts.T
        #print 'maximum tx is', np.max(t_x), ' and minimum is', np.min(t_x),'.....'
        t_x_diff = np.max(t_x) - np.min(t_x)
        t_y_diff = np.max(t_y) - np.min(t_y)
        t_z_diff = np.max(t_z) - np.min(t_z)
        # Calculate a sclaing factor for the subject to match tempalte size and avoid 'Nan' by zero division
        if t_x_diff == 0 and s_x_diff == 0:
          x_scale = 0.
        else:
          x_scale = t_x_diff / s_x_diff
        if t_y_diff == 0 and s_y_diff == 0:
          y_scale = 0.
        else:
          y_scale = t_y_diff / s_y_diff
        if t_z_diff == 0 and s_z_diff == 0:
          z_scale = 0.
        else:
          z_scale = t_z_diff / s_z_diff
        # Find center of mass
        cm_s = np.mean(s_pts, axis=0)
        cm_t = np.mean(t_pts, axis=0)
        initial_transl = (cm_t - cm_s)
        # Perform the transformation of the initial transformation matrix
        init_trans = np.zeros([4, 4])
        init_trans[:3, :3] = rotation3d(0., 0., 0.) * [x_scale, y_scale, z_scale]
        init_trans[0,3] = initial_transl[0]
        init_trans[1,3] = initial_transl[1]
        init_trans[2,3] = initial_transl[2]
        init_trans[3,3] = 1.
          
        def find_min(s_pts, init_trans):
          sourr = template_spacing /1e3
          auto_match_iters = np.array([ [0., 0., 0.],
                [0., 0., sourr], [0., 0., sourr * 2], [0., 0., sourr * 3],
                [sourr, 0., 0.], [sourr * 2, 0., 0.], [sourr * 3, 0., 0.],
                [0., sourr, 0.], [0., sourr * 2, 0.], [0., sourr * 3, 0.],
                [0., 0., -sourr], [0., 0., -sourr * 2], [0., 0., -sourr * 3],
                [-sourr, 0., 0.], [-sourr * 2, 0., 0.], [-sourr * 3, 0., 0.],
                [0., -sourr, 0.], [0., -sourr * 2, 0.], [0., -sourr * 3, 0.]])
              
          poss_trans = []
          for p, i in enumerate(auto_match_iters):
            # initial translation value
            tx, ty, tz = init_trans[0,3]+i[0], init_trans[1, 3]+i[1], init_trans[2, 3]+i[2]
            sx, sy, sz = init_trans[0,0], init_trans[1,1], init_trans[2,2]
            rx, ry, rz = 0, 0, 0
            x0 = (tx, ty, tz, rx, ry, rz)
              
            def error(x):
              tx, ty, tz, rx, ry, rz = x # 
              trans0 = np.zeros([4, 4])
              trans0[:3, :3] = rotation3d(rx, ry, rz) * [sx, sy, sz]
              trans0[0,3] = tx
              trans0[1,3] = ty
              trans0[2,3] = tz
              # rot and sca
              est = np.dot(s_pts, trans0[:3,:3].T)
              # transl
              est += trans0[:3, 3]
              if e_func == 'balltree':
                err = errfunc(est[:, :3], temp_tree)
              elif e_func == 'euclidean':
                err = errfunc(est[:, :3], t_pts)
              return err
                  
            est, _, info, msg, _ = leastsq(error, x0, full_output=True)
            est = np.concatenate((est, (init_trans[0,0], init_trans[1,1], init_trans[2,2]) ))
            trans = _trans_from_est(est)
            poss_trans.append(trans) 
              
          return (poss_trans)
          
      # Find the min summed distance for initial transformation
      poss_trans = find_min(s_pts, init_trans)
      #all_dist_sum_l = []
      all_dist_max_l = []
      all_dist_mean_l = []
      all_dist_var_l = []
      all_dist_err_l = []
      for tra in poss_trans:
        to_match_points = s_pts
        to_match_points = apply_trans(tra, to_match_points)
        if e_func == 'balltree':
          #all_dist_sum_l.append( np.array( ( np.sum( errfunc(to_match_points[:, :3], temp_tree) ) ) ) )
          all_dist_max_l.append( np.array( ( np.max( errfunc(to_match_points[:, :3], temp_tree) ) ) ) )
          all_dist_mean_l.append( np.array( ( np.mean( errfunc(to_match_points[:, :3], temp_tree) ) ) ) )
          all_dist_var_l.append( np.array( ( np.var( errfunc(to_match_points[:, :3], temp_tree) ) ) ) )
          all_dist_err_l.append(np.array( errfunc(to_match_points[:, :3], temp_tree)) )
        if e_func == 'euclidean':
          #all_dist_sum_l.append( np.array( ( np.sum( errfunc(to_match_points[:, :3], t_pts) ) ) ) )
          all_dist_max_l.append( np.array( ( np.max( errfunc(to_match_points[:, :3], t_pts) ) ) ) )
          all_dist_mean_l.append( np.array( ( np.mean( errfunc(to_match_points[:, :3], t_pts) ) ) ) )
          all_dist_var_l.append( np.array( ( np.var( errfunc(to_match_points[:, :3], t_pts) ) ) ) )
          all_dist_err_l.append( np.array(errfunc(to_match_points[:, :3], t_pts)) )
        del to_match_points
      #all_dist_sum = np.asarray(all_dist_sum_l)
      all_dist_max = np.asarray(all_dist_max_l)
      all_dist_mean = np.asarray(all_dist_mean_l)
      all_dist_var = np.asarray(all_dist_var_l)
      # Decide for the bestg fitting Transformation-Matrix
      idx1 = np.where(all_dist_mean == np.min(all_dist_mean))[0][0]
      # Collect all the possible inidcator values
      trans = poss_trans[idx1]
      del poss_trans
      to_match_points = s_pts
      to_match_points = apply_trans(trans, to_match_points)
      if e_func == 'balltree':
        #all_dist_sum = np.array( ( np.sum( errfunc(to_match_points[:, :3], temp_tree) ) ) )
        all_dist_max = np.array( ( np.max( errfunc(to_match_points[:, :3], temp_tree) ) ) )
        all_dist_mean = np.array( ( np.mean( errfunc(to_match_points[:, :3], temp_tree) ) ) )
        all_dist_var = np.array( ( np.var( errfunc(to_match_points[:, :3], temp_tree) ) ) )
        all_dist_err = (errfunc(to_match_points[:, :3], temp_tree))
      if e_func == 'euclidean':
        #all_dist_sum = np.array( ( np.sum( errfunc(to_match_points[:, :3], t_pts) ) ) )
        all_dist_max = np.array( ( np.max( errfunc(to_match_points[:, :3], t_pts) ) ) )
        all_dist_mean = np.array( ( np.mean( errfunc(to_match_points[:, :3], t_pts) ) ) )
        all_dist_var = np.array( ( np.var( errfunc(to_match_points[:, :3], t_pts) ) ) )
        all_dist_err = (errfunc(to_match_points[:, :3], t_pts))
      del to_match_points
      # Append the Dictionaries with the result and error values
      label_trans_dic.update({label:trans})
      label_trans_dic_mean_dist.update({label:np.min(all_dist_mean)})
      label_trans_dic_max_dist.update({label:np.min(all_dist_max)})
      label_trans_dic_var_dist.update({label:np.min(all_dist_var)})
      label_trans_dic_err.update({label:all_dist_err})
    
    if save_trans:
      print '\n#### Attempting to create and write MatchMaking Transformations to file..'
      fname_lw_trans = subject_dir + subject_from + '/' + '%s_%s_lw-trans.npy' %(subject_from, subject_to)
      mat_mak_trans_dict = {}
      mat_mak_trans_dict['ID'] = '%s -> %s' %(subject_from, subject_to)
      mat_mak_trans_dict['Labeltransformation'] = label_trans_dic
      mat_mak_trans_dict['Transformation Error[mm]'] = label_trans_dic_err
      mat_mak_trans_dict['Mean Distance Error [mm]'] = label_trans_dic_mean_dist
      mat_mak_trans_dict['Max Distance Error [mm]'] = label_trans_dic_max_dist
      mat_mak_trans_dict['Distance Variance Error [mm]'] = label_trans_dic_var_dist
      mat_mak_trans_dict_arr = np.array([mat_mak_trans_dict])
      np.save(fname_lw_trans, mat_mak_trans_dict_arr)
      print '    [done]'
      
    time_calc = ((time.time() - start_time) / 60)
    print '\n    Calculation Time: %.2f minutes.\n\n' %(time_calc)
    
    print '\n    ######################################'
    print '####          LW Matching Function         ####'
    print '####               SUCCESFULL              ####'
    print '    ######################################'
  
    return (label_trans_dic, label_trans_dic_err, label_trans_dic_mean_dist,
            label_trans_dic_max_dist, label_trans_dic_var_dist)
  

def _transform_src_lw(vsrc_subject_from, label_list_subject_from,
                      volume_labels, subject_to,
                      subjects_dir):
    """
    Parameters
    ----------
    vsrc_subject_from : Vol. Source Space
          Vol. Source Space
    volume_labels : 
          List of the volume labels of intrest
    subject_to : str
          template subject name
    subject_dir : String
          subjects directory
  
    
    Returns
    -------
    transformed_p : array
          Transformed points from subject volume source space to volume source
          space of the template subject.
    idx_vertices : array
          Array of idxs for all transformed vertices in the volume source space.
    """

    print '\n#### Attempting to transform subject source space labelwise..'
    st_time = time.time()
    subj_vol = vsrc_subject_from
    subject = subj_vol[0]['subject_his_id']
    x, y, z = subj_vol[0]['rr'].T
    subj_p = np.c_[x, y, z]
    label_list = label_list_subject_from
    fname_aseg = subjects_dir + subject + '/mri/aseg.mgz'
    
    print '\n#### Attempting to read MatchMaking Transformations from file..'
    fname_lw_trans = subjects_dir + subject + '/%s_%s_lw-trans.npy' %(subject, subject_to)
    if os.path.exists(fname_lw_trans):
      print '    MatchMaking Transformations file found.'
      mat_mak_trans_dict_arr = np.load(fname_lw_trans)
      label_trans_ID = mat_mak_trans_dict_arr[0]['ID']
      print '    Reading MatchMaking file %s..' %label_trans_ID
      label_trans_dic = mat_mak_trans_dict_arr[0]['Labeltransformation']
      print '    [done]'
    else:
      print '    MatchMaking Transformations file NOT found\n'
      print """    Please calculate the according transformation matrix dictionary
          by using the jumeg.jumeg_volmorpher.automatch_labels function"""
        
                      # Gathering volume labels
    mgz = nib.load(fname_aseg)
    mgz_data = mgz.get_data()
    lut = _get_lut()
    vox2rastkr_trans = _get_mgz_header(fname_aseg)['vox2ras_tkr']
    vox2rastkr_trans[:3] /= 1000.
    inv_vox2rastkr_trans = linalg.inv(vox2rastkr_trans)
    all_volume_labels = []
    vol_lab = mne.get_volume_labels_from_aseg(fname_aseg)
    for lab in vol_lab: all_volume_labels.append( lab.encode() )
    
    print """\n#### Attempting to check for vertice duplicates in labels
       due to spatial aliasing in volume source creation.."""
    for p, label in enumerate(all_volume_labels):
      lab_arr = label_list[label]
      lab_id = _get_lut_id(lut, label, True)[0]
      loadingBar(p, len(all_volume_labels), task_part=None)
      del_ver_idx_list = []  
      for arr_id, i in enumerate(lab_arr, 0):
        lab_vert_coord = subj_vol[0]['rr'][i]
        lab_vert_mgz_idx = apply_trans(inv_vox2rastkr_trans, lab_vert_coord)
        orig_idx = mgz_data[int( round(lab_vert_mgz_idx[0]) ),
                            int( round(lab_vert_mgz_idx[1]) ),
                            int( round(lab_vert_mgz_idx[2]) )]
        if orig_idx != lab_id:
          del_ver_idx_list.append(arr_id)      
      del_ver_idx = np.asarray(del_ver_idx_list)
      label_list[label] = np.delete(label_list[label], del_ver_idx)
    vert_sum = []
    for i in volume_labels:
      vert_sum.append(label_list[i].shape[0])
      for j in volume_labels:
        if i != j:
          h = np.intersect1d(label_list[i], label_list[j])
          if h.shape[0] > 0:
            print 'In Label:', i, ' are vertices from Label:', j,'(', h.shape[0],')'
            break
    print '    Total number of vertices is:', np.sum(np.asarray(vert_sum)),'\n\n'
    
    print """\n#### Attempting to transform %s source space labelwise to %s
       source space..""" %(subject, subject_to)
    transformed_p = np.array([[0,0,0]])
    vert_sum = []
    idx_vertices = []
    for p, labels in enumerate(volume_labels):
      vert_sum.append(label_list[labels].shape[0])
      idx_vertices.append(label_list[labels])
      loadingBar(p, len(volume_labels), task_part=labels)
      trans_p  = subj_p[label_list[labels]]
      trans = label_trans_dic[labels]
      # apply trans
      trans_p = apply_trans(trans, trans_p)
      del trans
      transformed_p = np.concatenate((transformed_p, trans_p))
      del trans_p
    transformed_p = transformed_p[1:]
    idx_vertices = np.concatenate(np.asarray(idx_vertices))
    calc_time = ((time.time() - st_time) / 60)
    print '    [done]\n'
    print '    Calculation Time: %.2f minutes.\n\n' %(calc_time)
    
    return (transformed_p, idx_vertices)


def volume_morph_stc(fname_stc_orig, subject_from, fname_vsrc_subject_from,
                     volume_labels, subject_to, fname_vsrc_subject_to,
                     cond, n_iter, interpolation_method, normalize,
                     subjects_dir, save_stc=False):
    """ Perform a volume morphing from one subject to a template.
    
    Parameters
    ----------
    fname_stc_orig : str
          Filepath of the original stc
    subject_from : str
          Subject ID
    fname_vsrc_subject_from : str
          Filepath of the subjects volume source space
    volume_labels : list of volume Labels
          List of the volume labels ofr intrest
    subject_to : str
          The tempalte subjects ID
    fname_vsrc_subject_to : str
          Filepath of the template subjects volume source space
    interpolation_method : str | None
          Either 'balltree' or 'euclidean'
    subjects_dir : str | None
          Subjects directory
    save : Boolean
          True to save. False is default
  
    Returns
    -------
    new-data : dictionary of one or more new stc
          The generated source time courses.
    idx_vertices : arr of int
          Vertice indices for all vertices in the given labels
  
    """ 
    print '\n    ######################################'
    print '####                  START                ####'
    print '####             Volume Morphing           ####'
    print '    ######################################'
    
    print '\n#### Attempting to read essential data files..'
    # STC 
    stc_orig = mne.read_source_estimate(fname_stc_orig) # note: vertices are of shape 1D
    stcdata = stc_orig.data
    nvert, ntimes = stc_orig.shape
    tmin, tstep = stc_orig.times[0], stc_orig.tstep
        
    # Source Spaces
    subj_vol = mne.read_source_spaces(fname_vsrc_subject_from)
    temp_vol = mne.read_source_spaces(fname_vsrc_subject_to)
    fname_subj_aseg = subjects_dir + subject_from + '/mri/aseg.mgz'
    fname_label_list_subject_from = fname_vsrc_subject_from[:-4] + '_vertno_labelwise.npy'
    label_list_subject_from = np.load(fname_label_list_subject_from).item()
    fname_label_list_subject_to = fname_vsrc_subject_to[:-4] + '_vertno_labelwise.npy'
    label_list_subject_to = np.load(fname_label_list_subject_to).item()
    print '    [done]'
    
    # Check for vertice duplicates
    mgz = nib.load(fname_subj_aseg)
    mgz_data = mgz.get_data()
    lut = _get_lut()
    vox2rastkr_trans = _get_mgz_header(fname_subj_aseg)['vox2ras_tkr']
    vox2rastkr_trans[:3] /= 1000.
    inv_vox2rastkr_trans = linalg.inv(vox2rastkr_trans)
    all_volume_labels = []
    vol_lab = mne.get_volume_labels_from_aseg(fname_subj_aseg)
    for lab in vol_lab: all_volume_labels.append( lab.encode() )
    print """\n#### Attempting to check for vertice duplicates in labels
     due to spatial aliasing in volume source creation.."""
    for p, label in enumerate(all_volume_labels):
      lab_arr = label_list_subject_from[label]
      lab_id = _get_lut_id(lut, label, True)[0]
      loadingBar(p, len(all_volume_labels), task_part=None)
      del_ver_idx_list = []  
      for arr_id, i in enumerate(lab_arr, 0):
        lab_vert_coord = subj_vol[0]['rr'][i]
        lab_vert_mgz_idx = apply_trans(inv_vox2rastkr_trans, lab_vert_coord)
        orig_idx = mgz_data[int( round(lab_vert_mgz_idx[0]) ),
                            int( round(lab_vert_mgz_idx[1]) ),
                            int( round(lab_vert_mgz_idx[2]) )]
        if orig_idx != lab_id:
          del_ver_idx_list.append(arr_id)      
      del_ver_idx = np.asarray(del_ver_idx_list)
      label_list_subject_from[label] = np.delete(label_list_subject_from[label],
                             del_ver_idx)
          
    vert_sum = []
    for i in volume_labels:
      vert_sum.append(label_list_subject_from[i].shape[0])
      for j in volume_labels:
        if i != j:
          h = np.intersect1d(label_list_subject_from[i],
                             label_list_subject_from[j])
          if h.shape[0] > 0:
            print 'In Label:', i, ' are vertices from Label:', j,'(', h.shape[0],')'
            break
    print '    Total number of vertices is:', np.sum(np.asarray(vert_sum))

    # =========================================================================
    #         Calculate the Transformation Matrixes
    # =========================================================================
    # Transform the whole subject source space lebelwise
    transformed_p, idx_vertices = _transform_src_lw(subj_vol,
                                                    label_list_subject_from,
                                                    volume_labels, subject_to,
                                                    subjects_dir)
    xn, yn, zn = transformed_p.T
      
    stcdata_sel = []
    for p, i in enumerate(idx_vertices):
      stcdata_sel.append( np.where( idx_vertices[p] == subj_vol[0]['vertno'] ) )
    stcdata_sel = np.asarray(stcdata_sel).flatten()
    stcdata_ch = stcdata[stcdata_sel]

    # =========================================================================
    #        Interpolate the data
    # =========================================================================     
    new_data = {}
    for inter_m in interpolation_method:

      print '\n#### Attempting to interpolate STC Data for every time sample..'
      print '    Interpolationmethod: ', inter_m
      st_time = time.time()
      xt, yt, zt = temp_vol[0]['rr'][temp_vol[0]['inuse'].astype(bool)].T  
      inter_data = np.zeros([xt.shape[0], ntimes])
      for i in range(0, ntimes):
        loadingBar(i, ntimes, task_part='Time sample: %i' %(i+1))
        inter_data[:,i] = griddata( (xn, yn, zn),
                  stcdata_ch[:,i], (xt, yt, zt),
                  method=inter_m, rescale=True)
      if inter_m == 'linear':
        inter_data = np.nan_to_num(inter_data)
      
      if normalize:
        print '\n#### Attempting to normalize the vol-morphed stc..'
        normalized_new_data = inter_data.copy()
        for p, labels in enumerate(volume_labels):
          lab_verts = label_list_subject_from[labels]
          lab_verts_temp = label_list_subject_to[labels]
          subj_vert_idx = np.array([], dtype=int)
          for i in xrange(0, lab_verts.shape[0]):
            subj_vert_idx = np.append(subj_vert_idx,
                                      np.where(lab_verts[i]==subj_vol[0]['vertno'])
                                      )
          temp_vert_idx = np.array([], dtype=int)
          for i in xrange(0, lab_verts_temp.shape[0]):
            temp_vert_idx = np.append(temp_vert_idx,
                                      np.where(lab_verts_temp[i]==temp_vol[0]['vertno'])
                                      )
          a = np.sum(stc_orig.data[subj_vert_idx], axis=0)
          b = np.sum(inter_data[temp_vert_idx], axis=0)
          norm_m_score = a / b
          normalized_new_data[temp_vert_idx] *= norm_m_score

        new_data.update({inter_m +'_norm':normalized_new_data})
      else:
        new_data.update({inter_m:inter_data})
      calc_time = ((time.time() - st_time) / 60)
      print '    Calculation Time: %.2f minutes.' %(calc_time)
    if save_stc:
      print '\n#### Attempting to write interpolated STC Data to file..'
      for inter_m in interpolation_method:
        fname_stc_orig_moprhed = fname_stc_orig[:-7] + '_morphed_to_%s_%s-vl.stc' %(subject_to, inter_m)
        print '    Destination:', fname_stc_orig_moprhed
        if normalize:
          _write_stc(fname_stc_orig_moprhed, tmin=tmin, tstep=tstep,
                     vertices=temp_vol[0]['vertno'], data=new_data[inter_m+'_norm'])
          _volumemorphing_plot_results(stc_orig, new_data[inter_m+'_norm'],
                                       interpolation_method,
                                       subj_vol, label_list_subject_from,
                                       temp_vol, label_list_subject_to,
                                       volume_labels, subject_from,
                                       subject_to, cond=cond, n_iter=n_iter,
                                       subjects_dir=subjects_dir)
        else:
          _write_stc(fname_stc_orig_moprhed, tmin=tmin, tstep=tstep,
                     vertices=temp_vol[0]['vertno'], data=new_data[inter_m])
          _volumemorphing_plot_results(stc_orig, new_data[inter_m],
                                       interpolation_method,
                                       subj_vol,
                                       temp_vol,
                                       volume_labels, subject_from,
                                       subject_to, cond=cond, n_iter=n_iter,
                                       subjects_dir=subjects_dir)
      print '[done]'
      new_data = mne.read_source_estimate(fname_stc_orig_moprhed)
      print '\n    ######################################'
      print '####             Volume Morphing           ####'
      print '####                SUCCESFULL             ####'
      print '    ######################################'
      
      return new_data
    else:
      print '#### Volume morphed stc data NOT saved..'
      
    print '\n    ######################################'
    print '####             Volume Morphing           ####'
    print '####                SUCCESFULL             ####'
    print '    ######################################'
      
    return new_data


def _volumemorphing_plot_results(stc_orig, new_data,
                                interpolation_method,
                                volume_orig, label_list_from, 
                                volume_temp, label_list_to,
                                volume_labels, subject, subject_to,
                                cond, n_iter,
                                subjects_dir):
    """ 
    Parameters
    ----------
    
    
    Returns
    -------
    
    """
    if subject_to is None:
      subject_to = ''
    else:
      subject_to = subject_to
    if cond is None:
      cond = ''
    else:
      cond = cond
    if n_iter is None:
      n_iter = ''
    else:
      n_iter = n_iter
      
    subj_vol = volume_orig
    subject_from = volume_orig[0]['subject_his_id']
    temp_vol = volume_temp
    subject_to = volume_temp[0]['subject_his_id']
    label_list = label_list_from
    label_list_template = label_list_to
    
    print '\n#### Attempting to save the volume morphing results ..'
    directory = subjects_dir + '%s/plots/VolumeMorphing/' %(subject)
    if not os.path.exists(directory):
      os.makedirs(directory)
      
    print """\n#### Attempting to compare subjects activity and interpolated
    activity in template for all labels.."""
    subj_lab_act = {}
    temp_lab_act = {}
    for p, label in enumerate(volume_labels):
      lab_arr = label_list[str(label)]
      lab_arr_temp = label_list_template[str(label)]
      subj_vert_idx = np.array([], dtype=int)
      temp_vert_idx = np.array([], dtype=int)
      for i in xrange(0, lab_arr.shape[0]):
        subj_vert_idx = np.append(subj_vert_idx, np.where(lab_arr[i] == subj_vol[0]['vertno']) )
      for i in xrange(0, lab_arr_temp.shape[0]):
        temp_vert_idx = np.append(temp_vert_idx, np.where(lab_arr_temp[i] == temp_vol[0]['vertno']) )
      lab_act_sum = np.array([])
      lab_act_sum_temp = np.array([])

      for t in xrange(0, stc_orig.times.shape[0]):
        lab_act_sum = np.append(lab_act_sum,np.sum(stc_orig.data[subj_vert_idx, t]))
        lab_act_sum_temp = np.append(lab_act_sum_temp, np.sum(new_data[temp_vert_idx, t]))
      subj_lab_act.update({label:lab_act_sum})
      temp_lab_act.update({label:lab_act_sum_temp})
                 
    data_dist = []
    for lab in volume_labels:
      a = np.asarray( subj_lab_act[str(lab)] )
      b = np.asarray( temp_lab_act[str(lab)] )
      data_dist.append( b - a )
    data_dist_arr = np.asarray(data_dist)
    
    label_length = np.linspace(1, len(volume_labels), len(volume_labels))          
    vmin = np.min([np.min(data_dist_arr)] )
    vmax = np.max([np.max(data_dist_arr)] ) 
    lim = np.max(np.abs([vmin, vmax]))
                    
    plt.figure(figsize=(18, 9))
    plt.subplot(1, 1, 1)
    im = plt.imshow(data_dist_arr, vmin=-lim, vmax=lim, cmap=cm.seismic,
                    extent=[stc_orig.times[0], stc_orig.times[-1], 0, 21],
                    aspect='auto' )
    plt.title('Amplitude Difference of Original (%s) and Interpolated Data' %(subject_from) )
    plt.yticks(label_length, volume_labels[::-1], rotation='horizontal')
    plt.xlabel('times [ms]')
    plt.tight_layout()
    cbar = plt.colorbar(im)
    cbar.ax.set_ylabel('Amplitude Difference []', rotation=270)
    cbar.ax.get_yaxis().labelpad = 15
    fname_save_fig = directory + '/%s_%s_%s_%s_labelwise-Difference_Amplitude' %(subject_from, subject_to, cond, n_iter)
    plt.savefig(fname_save_fig)
    plt.close()
    
    orig_act_sum = np.sum( stc_orig.data.sum(axis=0) )
    morphed_act_sum = np.sum( new_data.sum(axis=0) )
    act_diff_perc = ((morphed_act_sum - orig_act_sum) / orig_act_sum) * 100
    act_sum_morphed_normed = np.sum( new_data.sum(axis=0) )
    act_diff_perc_morphed_normed = ((act_sum_morphed_normed - orig_act_sum) / orig_act_sum) * 100
    
    f, (ax1) = plt.subplots(1, figsize=(16,5))
    ax1.plot(stc_orig.times, stc_orig.data.sum(axis=0), '#00868B',
             label='%s'%(subject_from) )
    ax1.plot(stc_orig.times, new_data.sum(axis=0), '#CD7600',
             label='%s morphed' %(subject_from) )
    ax1.plot(stc_orig.times, new_data.sum(axis=0), '#cdaa00',
             label='%s morphed + norm' %(subject_from) ) 
    ax1.set_title('Summed Source Amplitude - %s morphed on %s' %(subject_from, subject_to))
    ax1.text(stc_orig.times[0],
             np.maximum(stc_orig.data.sum(axis=0), new_data.sum(axis=0)).max(),
    """Total Amplitude Difference: %+.2f %%
    Total Amplitude Difference (norm):  %+.2f %%"""
    %(act_diff_perc, act_diff_perc_morphed_normed),
        size=12, ha="left", va="top",
        bbox=dict(boxstyle="round",
                  ec=("grey"),
                  fc=("white"),
                  )
        )
    ax1.set_ylabel('Summed Source Amplitude')
    ax1.legend(fontsize='large', facecolor="white", edgecolor="grey")
    ax1.get_xaxis().grid(True)
    plt.tight_layout()
    fname_save_fig = directory + '/%s_%s_%s_%s_labelwise-Amplitude' %(subject_from, subject_to, cond, n_iter)
    plt.savefig(fname_save_fig)
    plt.close()
    
    return


def plot_vstc(vstc, vsrc, tstep, subjects_dir, time_sample=None, coords=None, 
              save=False):
  """
  Parameters
  ----------
  fname_stc_orig : String
        Filename
  subject_from : list of Labels
        Filename
  Returns
  -------
  new-data : dictionary of one or more new stc
        The generated source time courses.
  """ 
  print '\n#### Attempting to plot volume stc from file..'
  print '    Creating 3D image from stc..'
  
  vstcdata = vstc.data
  img = vstc.as_volume(vsrc, dest='mri', mri_resolution=False)
  subject = vsrc[0]['subject_his_id']
  if vstc == 0:
    if tstep is not None:
      img = _make_image(vstc, vsrc, tstep, dest='mri', mri_resolution=False)
    else:
      print '    Please provide the tstep value !'
  print '    [done]'
  img_data = img.get_data()
  aff = img.affine
  if time_sample is None:
    print '    Searching for maximal Activation..'
    t = int(np.where(np.sum(vstcdata,axis=0)==np.max(np.sum(vstcdata,axis=0)))[0]) # maximum amp in time
  else:
    print '    using time sample', time_sample
    t = time_sample
  t_in_ms = vstc.times[t] * 1e3
  print '    Found time point: ', t_in_ms, 'ms'
  if coords is None:
    cut_coords = np.where( img_data == img_data[:,:,:,t].max() )
    print '    Respective Space Coords [meg-space]:'
    print '    X: ', cut_coords[0],'    Y: ', cut_coords[1],'    Z: ', cut_coords[2]
    max_try = np.concatenate( ( np.array([cut_coords[0][0]]), np.array([cut_coords[1][0]]), np.array([cut_coords[2][0]]) )  )
    cut_coords = apply_affine( aff, max_try )
  else:
    cut_coords = coords
  slice_x, slice_y, slice_z = int(cut_coords[0]), int(cut_coords[1]), int(cut_coords[2])
  print '    Respective Space Coords [mri-space]:'
  print '    X: ', slice_x,'    Y: ',slice_y,'    Z: ', slice_z
  temp_t1_fname = subjects_dir + subject +'/mri/T1.mgz'
  plt.figure(figsize=(16, 9))
  plotting.plot_stat_map( index_img( img, float(t) ), temp_t1_fname,
                                     figure=1,
                                     display_mode = 'ortho',
                                     threshold= vstcdata.min(),
                                     annotate=True,
                                     title='%s | t=%.2f ms'
                                     % (subject, t_in_ms),
                                     cut_coords=(slice_x, slice_y, slice_z),
                                     cmap='cold_hot' )
  if save:
    plt.savefig(subjects_dir + subject + '/plots/%s_vol-source-estimate_plot' %subject)
    plt.close()

  return


def _make_image(stc_data, src, tstep, dest='mri', mri_resolution=False):
    """Make a volume source estimate in a NIfTI file.

    Parameters
    ----------
    stc_data : Data of VolSourceEstimate
        The source estimate data
    src : list
        The list of source spaces (should actually be of length 1)
    tstep : float
        The tstep value for the recorded data
    dest : 'mri' | 'surf'
        If 'mri' the volume is defined in the coordinate system of
        the original T1 image. If 'surf' the coordinate system
        of the FreeSurfer surface is used (Surface RAS).
    mri_resolution: bool
        It True the image is saved in MRI resolution.
        WARNING: if you have many time points the file produced can be
        huge.

    Returns
    -------
    img : instance Nifti1Image
        The image object.
    """
    n_times = stc_data.shape[1]
    shape = src[0]['shape']
    shape3d = (shape[2], shape[1], shape[0])
    shape = (n_times, shape[2], shape[1], shape[0])
    vol = np.zeros(shape)
    mask3d = src[0]['inuse'].reshape(shape3d).astype(np.bool)

    if mri_resolution:
        mri_shape3d = (src[0]['mri_height'], src[0]['mri_depth'],
                       src[0]['mri_width'])
        mri_shape = (n_times, src[0]['mri_height'], src[0]['mri_depth'],
                     src[0]['mri_width'])
        mri_vol = np.zeros(mri_shape)
        interpolator = src[0]['interpolator']

    for k, v in enumerate(vol):
        v[mask3d] = stc_data[:, k]
        if mri_resolution:
            mri_vol[k] = (interpolator * v.ravel()).reshape(mri_shape3d)

    if mri_resolution:
        vol = mri_vol

    vol = vol.T

    if mri_resolution:
        affine = src[0]['vox_mri_t']['trans'].copy()
    else:
        affine = src[0]['src_mri_t']['trans'].copy()
    if dest == 'mri':
        affine = np.dot(src[0]['mri_ras_t']['trans'], affine)
    affine[:3] *= 1e3

    try:
        import nibabel as nib  # lazy import to avoid dependency
    except ImportError:
        raise ImportError("nibabel is required to save volume images.")

    header = nib.nifti1.Nifti1Header()
    header.set_xyzt_units('mm', 'msec')
    header['pixdim'][4] = 1e3 * tstep
    img = nib.Nifti1Image(vol, affine, header=header)
    
    return img

  
#%% ===========================================================================
# # Statistical Analysis Section
# =============================================================================

def sum_up_vol_cluster(clu, p_thresh=0.05, tstep=1e-3, tmin=0,
                           subject=None, vertices=None):
    T_obs, clusters, clu_pvals, _ = clu
    n_times, n_vertices = T_obs.shape
    good_cluster_inds = np.where(clu_pvals < p_thresh)[0]
    #  Build a convenient representation of each cluster, where each
    #  cluster becomes a "time point" in the VolSourceEstimate
    if len(good_cluster_inds) > 0:
        data = np.zeros((n_vertices, n_times))
        data_summary = np.zeros((n_vertices, len(good_cluster_inds) + 1))
        print 'Data_summary is in shape of:', data_summary.shape
        for ii, cluster_ind in enumerate(good_cluster_inds):
            loadingBar(ii+1, len(good_cluster_inds), task_part='Cluster Idx %i' %(cluster_ind))
            data.fill(0)
            v_inds = clusters[cluster_ind][1]
            t_inds = clusters[cluster_ind][0]
            data[v_inds, t_inds] = T_obs[t_inds, v_inds]
            # Store a nice visualization of the cluster by summing across time
            data = np.sign(data) * np.logical_not(data == 0) * tstep
            data_summary[:, ii + 1] = 1e3 * np.sum(data, axis=1)
            # Make the first "time point" a sum across all clusters for easy
            # visualization
        data_summary[:, 0] = np.sum(data_summary, axis=1)

        return VolSourceEstimate(data_summary, vertices, tmin=tmin, tstep=tstep,
                              subject=subject)
    else:
        raise RuntimeError('No significant clusters available. Please adjust '
                           'your threshold or check your statistical '
                           'analysis.')
        
        
def plot_T_obs(T_obs, threshold, tail, save, fname_save):
    """ Visualize the Volume Source Estimate as an Nifti1 file """ 
    #T_obs plot code
    T_obs_flat = T_obs.flatten()
    plt.figure('T-Statistics', figsize=( (8,8) ))
    T_max = T_obs.max()
    T_min = T_obs.min()
    T_mean = T_obs.mean()
    str_tail = 'one tail'
    if tail is 0 or tail is None:
      plt.xlim([-20, 20])
      str_tail = 'two tail'
    elif tail is -1:
      plt.xlim([-20, 0])
    else:
      plt.xlim([0, 20])
    y, binEdges = np.histogram(T_obs_flat, bins=100)
    if threshold is not None:
      plt.plot([threshold, threshold], (0,y.max()), color='#CD7600',
               linestyle=':', linewidth=2)
    bincenters = 0.5 * (binEdges[1:] + binEdges[:-1])
    legend = """T-Statistics:
      Mean:  %.2f
      Minimum:  %.2f
      Maximum:  %.2f
      Threshold:  %.2f  
      """ % (T_mean, T_min, T_max, threshold)
    plt.xlabel('T-scores', fontsize=15)
    plt.ylabel('T-values count', fontsize=15)
    plt.title('T statistics distribution of t-test - %s' %str_tail, fontsize=16)
    plt.plot(bincenters, y, label=legend, color='#00868B')
#    plt.xlim([])
    plt.tight_layout()
    legend = plt.legend(loc='upper right', shadow=True, fontsize='large', frameon= True)
    plt.show()
    if save:
      plt.savefig(fname_save)
      plt.close()
    
    return
  

def plot_T_obs_3D(T_obs, save, fname_save):
    """ Visualize the Volume Source Estimate as an Nifti1 file """ 
    fig = plt.figure(facecolor='w', figsize=( (8, 8) ))
    ax = fig.gca(projection='3d')
    vertc, timez = np.mgrid[0:T_obs.shape[0], 0:T_obs.shape[1]]
    Ts = T_obs
    title = 'T Obs'
    t_obs_stats = ax.plot_surface(vertc, timez, Ts, cmap=cm.hot)#, **kwargs)
    #plt.set_xticks([])
    #plt.set_yticks([])
    ax.set_xlabel('times [ms]')
    ax.set_ylabel('Vertice No')
    ax.set_zlabel('Statistical Amplitude')
    ax.w_zaxis.set_major_locator(LinearLocator(6))
    ax.set_zlim(0, np.max(T_obs))
    ax.set_title(title)
    fig.colorbar(t_obs_stats, shrink=0.5)
    plt.tight_layout()
    plt.show()
    if save:
      plt.savefig(fname_save)
      plt.close()
    return
  