'''
;====================================================================================================
;                              X2ABUNDANCE
;
; Package for determining elemental weight percentages from XRF line fluxes
;
; Algorithm developed by P. S. Athiray (Athiray et al. 2015)
; Codes in IDL written by Netra S Pillai
; Codes for XSPEC localmodel developed by Ashish Jacob Sam and Netra S Pillai
;
; Developed at Space Astronomy Group, U.R.Rao Satellite Centre, Indian Space Research Organisation
;
;====================================================================================================

This file contains the function get_constants_xrf that interpolates the cross-sections from the database to the input energy axis and also takes into account inter-element effects
'''

from common_modules import *
from scipy.interpolate import interp1d
import numpy as np
from scipy.interpolate import interp1d

def get_constants_xrf(energy: list, at_no: list, weight: list, xrf_lines: Xrf_Lines) -> Const_Xrf:
    # Function to compute the different cross sections necessary for computing XRF lines

    totalweight = np.sum(weight)
    weight = weight / totalweight
    no_elements = n_elements(at_no)
    n_ebins = n_elements(energy)
    
    tmp2 = xrf_lines.edgeenergy
    n_lines = np.shape(tmp2)[1]
    
    musampletotal_echarline = np.zeros((no_elements, n_lines))
    musampletotal_eincident = np.zeros((no_elements, n_lines, n_ebins))
    muelementphoto_eincident = np.zeros((no_elements, n_lines, n_ebins))
    
    for i in range(no_elements):
        for j in range(n_lines):
            line_energy = xrf_lines.lineenergy[i, j]
            rad_rate = xrf_lines.radrate[i, j]
            edge_energy = xrf_lines.edgeenergy[i, j]
            if line_energy > 0 and rad_rate > 0:
                for k in range(no_elements):
                    # print(xrf_lines.energy_nist)
                    tmp3 = np.where(xrf_lines.energy_nist[k, :] != 0)
                    x_interp = (xrf_lines.energy_nist[k, tmp3])[0, :]
                    y_interp_total = (xrf_lines.totalcs_nist[k, tmp3])[0, :]
                    y_interp_photo = (xrf_lines.photoncs_nist[k, tmp3])[0, :]
                    
                    func_interp_total = interp1d(x_interp, y_interp_total, fill_value='extrapolate')
                    func_interp_photo = interp1d(x_interp, y_interp_photo, fill_value='extrapolate')
                        
                    muelement_echarline = func_interp_total(line_energy)
                    musampletotal_echarline[i, j] += weight[k] * muelement_echarline
                    
                    muelement_eincident = func_interp_total(energy)
                    musampletotal_eincident[i, j, :] += weight[k] * muelement_eincident
                    
                    muelementphoto_eincident[i, j, :] = func_interp_photo(energy)
                
                tmp4 = np.where(energy < edge_energy)
                if n_elements(tmp4) != 0:
                    musampletotal_eincident[i, j, tmp4] = 0.0
                    muelementphoto_eincident[i, j, tmp4] = 0.0
    
    return Const_Xrf(musampletotal_echarline, musampletotal_eincident, muelementphoto_eincident)