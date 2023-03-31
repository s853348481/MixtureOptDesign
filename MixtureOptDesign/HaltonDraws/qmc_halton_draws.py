from halton_draws import *
from scipy.stats import qmc

class QMCHaltonDraws(HaltonDraws):
    """
    Subclass to generate Halton draws using qmc.halton.
    """
    def generate_draws(self):
        """
        Generate Halton draws from a multi-normal distribution with a specified mean vector and covariance matrix
        using qmc.halton.

        Returns
        -------
        numpy.ndarray
            Halton draws from the multi-normal distribution.
            """
        
        # Generate Halton sequences
        sampler = qmc.Halton(d=self.beta.size, scramble=True)
        sample = sampler.random(n=self.n)
        
        # Apply inverse CDF of standard normal distribution
        norm_sequences = norm.ppf(sample)
        
        # Use predefined covariance matrix to transform standard normal random numbers to correlated normal random numbers
        correlated_norm_sequences = np.dot(np.linalg.cholesky(self.cov_matrix), norm_sequences.T)
        
        # Add mean vector to obtain final Halton draws
        self.halton_draws = correlated_norm_sequences.T + self.beta
        return self.halton_draws