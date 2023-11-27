import random
import numpy as np

def generate_random_numbers(count):
    """Generate a list of random numbers."""
    random_numbers = []
    for _ in range(count):
        random_numbers.append(random.randint(1, 100))  # Generates random integers between 1 and 100 (inclusive)
    return random_numbers

def generate_random_numbers1(count):
    """Generate a list of random numbers."""
    random_numbers = []
    for _ in range(count):
        random_numbers.append(9)  # Generates random integers between 1 and 100 (inclusive)
    return random_numbers

def generate_random_numbers_numpy(count):
    random_numbers = np.random.randint(1, 100, size=count)  # Generates 'count' random integers between 1 and 100
    return random_numbers.tolist()