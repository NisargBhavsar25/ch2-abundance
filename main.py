import sys

from scripts.abundance import Models
from scripts.utils import catalog, lunar_map
from scripts.flare_catalog import pipeline


def main():
    model = sys.argv[1]

    if model == "preprocess":
        '''
        Preprocess the data

        Usage: python main.py preprocess [subroutine]

        Subroutines:
        -----------
        - Select: select candidate CLASS files for XRF line catalog: python main.py preprocess select
            > Store the XSM zip files corresponding to the duration of the CLASS files in the preprocess/XSM_Data folder
            > Store the CLASS file names in the preprocess/classnames.txt file
            > Outputs the flare class of the CLASS files in the preprocess/output.csv file
        '''

        if len(sys.argv) != 3:
            print("Usage: python main.py preprocess [subroutine]")
            sys.exit()

        if sys.argv[2] == "select":
            pipeline.run()
        else:
            print("Invalid subroutine")
            sys.exit()

    filename = sys.argv[2]

    model = Models(model)
    model.load_data(filename)
    ratios = model.find_abundance()
    coordinates = model.get_coordinates()

    catalog.update_catalog(filename, coordinates, ratios)
    lunar_map.generate_surface_embedding(ratios, coordinates)


if __name__ == "__main__":
    main()
