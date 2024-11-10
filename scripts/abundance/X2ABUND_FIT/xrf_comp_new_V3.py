from common_modules import *
import numpy as np
from scipy.interpolate import interp1d

def xrf_comp(energy: list, counts, i_angle, e_angle, at_no: list, weight: list, xrf_lines:Xrf_Lines, const_xrf:Const_Xrf) -> Xrf_Struc:
    no_elements = len(at_no)
    # print(energy,len(energy),"in xrfcomp")
    weight = np.array(weight) / np.sum(weight) 
    sin_i_angle = np.sin(i_angle * np.pi / 180)
    sin_e_angle = np.sin(e_angle * np.pi / 180)
    
    n_lines = xrf_lines.edgeenergy.shape[1]
    binsize = energy[1] - energy[0]
    
    primary_xrf = np.zeros((no_elements, n_lines))
    secondary_xrf = np.zeros((no_elements, n_lines))
    
    for i in range(no_elements):
        for j in range(n_lines):
            fluoryield = xrf_lines.fluoryield[i, j]
            radrate = xrf_lines.radrate[i, j]
            lineenergy = xrf_lines.lineenergy[i, j]
            element_jumpfactor = xrf_lines.jumpfactor[i, :]
            element_edgeenergy = xrf_lines.edgeenergy[i, :]
            
            if j <= 1:  
                ratio_jump = np.where(energy >= element_edgeenergy[j], 1.0 - 1.0 / element_jumpfactor[j], 0)
            else: 
                ratio_jump = np.where(energy > element_edgeenergy[1], 
                                      1.0 / np.prod(element_jumpfactor[1:j]) * (1.0 - 1.0 / element_jumpfactor[j]), 
                                      0)
                for k in range(2, j + 1):
                    mask = (energy < element_edgeenergy[k - 1]) & (energy > element_edgeenergy[k])
                    if mask.any():
                        ratio_jump[mask] = 1.0 / np.prod(element_jumpfactor[k:j]) * (1.0 - 1.0 / element_jumpfactor[j])

            if lineenergy > 0 and radrate > 0:
                musample_eincident = const_xrf.musampletotal_eincident[i, j, :]
                musample_echarline = const_xrf.musampletotal_echarline[i, j]
                muelement_eincident = const_xrf.muelementphoto_eincident[i, j, :]
                
                pxrf_denom = musample_eincident / sin_i_angle + musample_echarline / sin_e_angle
                pxrf_Q = weight[i] * muelement_eincident * fluoryield * radrate * ratio_jump
                primary_xrf[i, j] = (1.0 / sin_i_angle) * np.sum((pxrf_Q * counts * binsize) / (pxrf_denom + 1e-12))

                secondaries_index_2D = np.where(xrf_lines.edgeenergy < lineenergy)
                for k in range(len(secondaries_index_2D[0])):
                    i_secondary = secondaries_index_2D[0][k]
                    j_secondary = secondaries_index_2D[1][k]
                    
                    fluoryield_secondary = xrf_lines.fluoryield[i_secondary, j_secondary]
                    radrate_secondary = xrf_lines.radrate[i_secondary, j_secondary]
                    lineenergy_secondary = xrf_lines.lineenergy[i_secondary, j_secondary]
                    
                    element_jumpfactor_secondary = xrf_lines.jumpfactor[i_secondary, :]
                    if j_secondary <= 1:
                        ratio_jump_secondary = 1.0 - 1.0 / element_jumpfactor[j_secondary]
                    else:
                        ratio_jump_secondary = (1.0 / np.prod(element_jumpfactor_secondary[1:j_secondary]) 
                                                * (1.0 - 1.0 / element_jumpfactor_secondary[j_secondary]))

                    if lineenergy_secondary > 0 and radrate_secondary > 0:
                        musample_echarline_secondary = const_xrf.musampletotal_echarline[i_secondary, j_secondary]
                        muelement_eincident_secondary = const_xrf.muelementphoto_eincident[i_secondary, j_secondary, :]
                        
                        func_interp = interp1d(energy, muelement_eincident_secondary, fill_value='extrapolate')
                        muelement_pline_secondary = func_interp(lineenergy)
                        
                        with np.errstate(divide='ignore', invalid='ignore'):
                            term1 = (sin_i_angle / musample_eincident) * np.log(1 + np.where(musample_eincident > 0, musample_eincident / (sin_i_angle * musample_echarline), 0))
                            term2 = (sin_e_angle / musample_echarline_secondary) * np.log(1 + np.where(musample_echarline_secondary > 0, musample_echarline_secondary / (sin_e_angle * musample_echarline), 0))
                            L = 0.5 * (term1 + term2)

                        L[musample_eincident == 0] = 0

                        sxrf_denom = musample_eincident / sin_i_angle + musample_echarline_secondary / sin_e_angle
                        sxrf_Q = weight[i_secondary] * muelement_pline_secondary * fluoryield_secondary * radrate_secondary * ratio_jump_secondary
                        
                        secondary_contribution = (1.0 / sin_i_angle) * np.sum((counts * pxrf_Q * sxrf_Q * L * binsize) / (sxrf_denom + 1e-12))
                        if secondary_contribution > 0:
                            secondary_xrf[i_secondary, j_secondary] += secondary_contribution

    total_xrf = primary_xrf + secondary_xrf
    return Xrf_Struc(primary_xrf, secondary_xrf, total_xrf)
