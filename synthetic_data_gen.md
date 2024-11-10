# Synthetic Data Generation using Fundamental Parameters Method
This code simulates an X-ray fluorescence (XRF) spectrum by incorporating fundamental parameters and Gaussian broadening effects to model the physical characteristics of fluorescence emission. It defines a class, ⁠ XRFSimulator ⁠, to calculate primary fluorescence intensity using parameters such as element concentration, geometric factors, absorption coefficients, excitation factors, and sample mass thickness. The code further applies Gaussian broadening to account for energy resolution limitations, influenced by the Fano factor, electronic noise, and the energy-to-pair-creation relationship in silicon detectors. Simulations include contributions from characteristic X-ray lines (Kα and Kβ) for multiple elements, with their respective intensities determined by the fundamental parameter equations. A visualization of the spectrum is generated, displaying both the raw X-ray line intensities and the broadened spectrum. Detailed annotations mark individual peaks with their corresponding element and line type. This model offers a robust framework for analyzing XRF spectra, aiding in qualitative and quantitative assessment of elemental compositions under varying experimental conditions.

### Input Requirements

1. **Fano Factor**: 
   - A numerical value representing the Fano factor, which accounts for the statistical variance in charge generation in the detector. 
   - **Example**: `0.12`

2. **Incident Angle**: 
   - The angle (in degrees) at which the X-ray beam strikes the sample. 
   - **Example**: `45°`

3. **Detector Angle**: 
   - The angle (in degrees) between the sample surface and the detector. 
   - **Example**: `30°`

4. **Element-Specific Parameters**:
   - **Element Concentrations**: 
     - Proportions of each element in the sample, expressed as fractions.
     - **Example**: `0.20` for silicon
   - **Characteristic X-ray Lines**: 
     - A list of characteristic energies (in keV) for each element, along with their relative intensities.
     - **Example**: `1.74 keV` with intensity `1.0` for Si Kα

5. **Fundamental Parameters**:
   - **Geometric Factor**: 
     - A numerical value representing the geometric efficiency of the detector setup.
     - **Example**: `0.1`
   - **Absorption Coefficients**: 
     - A dictionary containing the absorption coefficients (⁠`mu_lambda`⁠, `mu_i`⁠) and fluorescence yield (`tau`) for the sample material.
   - **Excitation Factor**: 
     - A scalar representing the efficiency of X-ray excitation.
     - **Example**: `0.8`
   - **Mass Thickness**: 
     - The effective thickness of the sample material, expressed in grams per square centimeter.
     - **Example**: `0.01`

6. **Energy Points** (Optional): 
   - The number of energy points for generating the broadened spectrum.
   - **Default**: `1000`