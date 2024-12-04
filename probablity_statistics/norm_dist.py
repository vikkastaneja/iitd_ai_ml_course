import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def generate_normal_distribution(mean, standard_deviation, size):
    # Generate a normal distribution array
    data = np.random.normal(mean, standard_deviation, size)
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

# Define the parameters of the normal distribution
mu = 0  # Mean
sigma = 1  # Standard deviation

number_of_samples = 100000
data = generate_normal_distribution(mu, sigma, number_of_samples)
assert len(data) == number_of_samples
assert find_probability_using_cdf_and_plot(data, 0, 0, 1, False) == 0.5

# Generate a range of x values
x = np.linspace(-np.sort(data)[0], np.sort(data)[0], number_of_samples)

# Calculate the PDF values for each x value
y = norm.pdf(x, mu, sigma)
probability_left = norm.pdf(mu-sigma, mu, sigma)
probability_right = norm.pdf(mu+sigma, mu, sigma)

probability_left_left = norm.pdf(mu-2*sigma, mu, sigma)
probability_right_right = norm.pdf(mu+2*sigma, mu, sigma)

# Plot the normal distribution along with mu+-sigma and mu+-2*sigma
plt.plot(x, y)
plt.plot([mu-sigma, mu-sigma], [0, probability_left], 'g--')
plt.plot([mu+sigma, mu+sigma], [0, probability_right], 'g--')
plt.plot([np.sort(data)[0], mu+sigma], [probability_right, probability_right], 'g--')
plt.text(mu+sigma, probability_right + 0.02, f'({mu+sigma},{probability_right:.2f})', ha='center')
plt.text(mu-sigma, probability_right + 0.02, f'({mu-sigma},{probability_right:.2f})', ha='center')

plt.plot([mu-2*sigma, mu-2*sigma], [0, probability_left_left], 'p--')
plt.plot([mu+2*sigma, mu+2*sigma], [0, probability_right_right], 'p--')
plt.plot([np.sort(data)[0], mu+2*sigma], [probability_right_right, probability_right_right], 'p--')
plt.text(mu+2*sigma, probability_right_right + 0.02, f'({mu+2*sigma},{probability_right_right:.2f})', ha='center')
plt.text(mu-2*sigma, probability_right_right + 0.02, f'({mu-2*sigma},{probability_right_right:.2f})', ha='center')

plt.title(f"Normal Distribution (Mean={mu}, Standard Deviation={sigma})\nSample size={number_of_samples}")
plt.xlabel("---- x --->")
plt.ylabel("---- Probability --->")
plt.show(block=False)