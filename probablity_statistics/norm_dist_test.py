from norm_dist import generate_normal_distribution as generate_distribution
from norm_dist import find_probability_using_cdf_and_plot as find_probability_using_cdf

def test_generated_distribution():
    number_of_samples = 10000
    data = generate_distribution(0, 1, number_of_samples)
    assert len(data) == number_of_samples

def test_find_probability_of_mean():
    number_of_samples = 10000
    data = generate_distribution(0, 1, number_of_samples)
    assert find_probability_using_cdf(data, 0, 0, 1, False) == 0.5