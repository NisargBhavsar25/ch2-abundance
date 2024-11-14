import polars as pl # using polars
import numpy as np
import matplotlib.pyplot as plt

class KneePoint:
    def __init__(self, df):
        self.df = df
        self.df_sorted = None
        self.knee_x = None
        self.knee_y = None

    def calculate_knee_point(self, plot=True):
        # sort by distance and reset the rows
        self.df_sorted = self.df.sort('distance').with_row_index()

        y = self.df_sorted['distance'].to_numpy()
        x = self.df_sorted['index'].to_numpy()

        # define endpoints of the curve
        p1 = np.array([x[0], y[0]])
        p2 = np.array([x[-1], y[-1]])

        # vector calculations for perpendicular distances to the line
        line_vec = p2 - p1
        line_vec_norm = line_vec / np.linalg.norm(line_vec)

        distances = np.abs(np.cross(line_vec, np.c_[x - p1[0], y - p1[1]])) / np.linalg.norm(line_vec)

        # the knee point is the one with maximum distance
        knee_index = np.argmax(distances)
        self.knee_x = x[knee_index]
        self.knee_y = y[knee_index]

        if plot:
            self.plot_knee_point(x, y)

        print(f"Knee point at x = {self.knee_x}, y = {self.knee_y}\n" + "-"*100)
        return self.df_sorted, (self.knee_x, self.knee_y)

    def plot_knee_point(self, x, y):
        fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(9, 4))

        # original data plot
        plt1.plot(x, y)
        plt1.set_xlabel("Index")
        plt1.set_ylabel("Distances")
        plt1.set_title("Data Plot")

        # knee point plot
        plt2.plot(x, y)
        plt2.plot(self.knee_x, self.knee_y, 'rx')
        plt2.axvline(x=self.knee_x, color='r', linestyle='--')
        plt2.set_xlabel("Index")
        plt2.set_ylabel("Distances")
        plt2.set_title("Knee Point of Data")

        plt.show()

    def plot_samples(self, knee=None, before=False, after=False, top=False):
        knee_x, knee_y = knee if knee else (self.knee_x, self.knee_y)

        df_filtered = self.df_sorted.filter(pl.col("index") > knee_x)

        if before:
            print("100 samples just before the knee point")
            sample_data = self.df_sorted.slice(knee_x - 100, 100)
            self.draw_plots(sample_data)

        if after:
            print("100 samples just after the knee point")
            sample_data = df_filtered.slice(0, 100)
            self.draw_plots(sample_data)

        if top:
            print("Top 100 samples")
            sample_data = df_filtered.tail(100)
            self.draw_plots(sample_data)

        print("Random 100 samples after the knee point")
        sample_data = df_filtered.sample(n=100, seed=69)
        self.draw_plots(sample_data)

    def draw_plots(self, sample_data):
        # helper function for the plot_samples function
        fig, plots = plt.subplots(20, 5, figsize=(20, 80))
        plots = plots.flatten()

        for i, row in enumerate(sample_data.iter_rows(named=True)):
            plts = plots[i]
            plts.plot(row['x'], row['y'])
            plts.set_yscale('log')
            plts.set_xlim([1, 5])
            plts.set_title(f"{row['distance']}")
            plts.set_xlabel("x")
            plts.set_ylabel("y")

        plt.tight_layout()
        plt.show()
        print("-"*100)


# example usage
# df = pl.read_parquet()

# knee_finder = KneePoint(df)

# sorted_df, knee = knee_finder.calculate_knee_point(plot=True)

# knee_finder.plot_samples(knee=knee)