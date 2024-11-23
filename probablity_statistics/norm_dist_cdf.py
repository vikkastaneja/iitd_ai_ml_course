import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_normal_distribution(mean, standard_deviation, size):
    # Generate a normal distribution array
    mu = 0  # Mean
    sigma = 1  # Standard deviation

    data = np.random.normal(mu, sigma, size)
    return data

def find_probability_using_cdf_and_plot(data, value, mean, standard_deviation, block=False):

    # Calculate the CDF
    x = np.sort(data)
    y = norm.cdf(x, mean, standard_deviation)

    # Plot the CDF
    plt.plot(x, y)
    plt.title('CDF of a Normal Distribution')
    plt.xlabel('Value')
    plt.ylabel('Probability')

    # Mark a probability based on a value
    probability = norm.cdf(value, mean, standard_deviation)
    print("Probability", probability)
    plt.plot([value, value], [0, probability], 'r--')
    plt.plot([x[0], value], [probability, probability], 'b--')
    plt.text(value, probability + 0.02, f'P(X <= {value}) = {probability:.2f}', ha='center')

    plt.show(block=block)
    return probability


number_of_samples = 10000
data = generate_normal_distribution(0, 1, number_of_samples)
assert len(data) == number_of_samples
assert find_probability_using_cdf_and_plot(data, 0, 0, 1, False) == 0.5