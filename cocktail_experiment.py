from MixtureOptDesign.data.csv_util import read_csv_file
import numpy as np
from MixtureOptDesign import HierarchicalCluster, HaltonDraws,plot_ternary_design
from MixtureOptDesign.data.csv_util import read_csv_file
from MixtureOptDesign.CoordinateExchange.coordinate import ClusteredCoordinateExchangeIOptimal
import time



def main():
    # load beta halton draws
    data = np.genfromtxt("Tests/data/beta_mario.csv", delimiter=',')
    beta_ma = np.array(data)
    
    
    best_design_mario = read_csv_file("Tests/data/design_03.csv")
    initial_design = read_csv_file("Tests/data/initial_design_Mario_cocktail.csv")
    
    beta = np.array( (1.36, 1.57, 2.47, -0.43, 0.50, 1.09))
    
    sigma0 = np.array([[6.14, 5.00, 2.74, -0.43, -2.81, -3.33],
                   [5.00, 6.76, 4.47, -1.79, -6.13, -3.51],
                   [2.74, 4.47, 3.45, -1.38, -4.71, -2.17],
                   [-0.43, -1.79, -1.38, 1.18, 2.39, 0.71],
                   [-2.81, -6.13, -4.71, 2.39, 7.43, 2.71],
                   [-3.33, -3.51, -2.17, 0.71, 2.71, 2.49]])
    
    
    
    fig_initial_design = plot_ternary_design(initial_design)
    fig_best_design = plot_ternary_design(initial_design)
    
    fig_initial_design.write_image("MixtureOptDesign/data/initial_design_cocktail.png")
    fig_best_design.write_image("MixtureOptDesign/data/best_design_cocktail.png")
    
    # halt1 = HaltonDraws(beta,sigma0,128)
    # beta_draws = halt1.generate_draws()
    
    coord = ClusteredCoordinateExchangeIOptimal(num_ingredient=3, num_sets=2, num_choices=16, order=3, n_points=30, bayesian=True, beta=beta_ma, iteration=10, kappa=1, sigma=None)
    
    h_clust = HierarchicalCluster(best_design_mario,)

    h_clust.get_elbow_curve(beta_ma,3,"cocktail")
    
    # Define the cluster numbers to loop through
    cluster_nums = [10,9,8]
    
    cluster_metric = "average"
    start_time = time.perf_counter()
    # Loop through the cluster numbers
    for cluster_num in cluster_nums:

        # Fit the hierarchical clustering model
        h_clust.fit(cluster_num, cluster_metric )

        # Get the replaced data and reshape it
        replaced_data = h_clust.design()
        cluster_design = replaced_data.T.reshape(best_design_mario.shape)

        # Optimize the design and plot it
        optimal_design_clust = coord.optimize_design(design_=cluster_design)
        fig_optimal_design = plot_ternary_design(optimal_design_clust)

        # Save the plot as an image
        fig_optimal_design.write_image(f"MixtureOptDesign/data/{cluster_metric}_design_cocktail_{cluster_num}.png")
    end_time = time.perf_counter()
    print("Time taken:", end_time - start_time)



if __name__ == '__main__':
    main()