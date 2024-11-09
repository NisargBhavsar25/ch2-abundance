from astropy.io import fits

# # Load the RMF file
# rmf_file = "/home/manasj/Downloads/cla/miscellaneous/X2ABUND_LMODEL_V1/test/class_rmf_v1.rmf"  # Replace with your RMF file path
# with fits.open(rmf_file) as rmf:
#     # Check the contents
#     rmf.info()
#     # Access a specific HDU (Header Data Unit), often the RESPONSE matrix is in the second HDU
#     rmf_data = rmf[1].data  # Check which HDU contains the data you need
#     # Print columns
#     print(rmf_data.columns)

# Load the ARF file
arf_file = "./test/class_arf_v1.arf"  # Replace with your ARF file path
with fits.open(arf_file) as arf:
    # Check the contents
    arf.info()
    # Access the effective area data
    arf_data = arf[1].data  # Effective area data is usually in the second HDU
    # Print columns
    print(arf_data.columns)
