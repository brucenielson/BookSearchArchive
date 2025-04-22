import random
from math import comb, ceil


def probability_at_least_x_percent_heads(x_fraction, n_flips=10):
    k = ceil(x_fraction * n_flips)
    probability = sum(comb(n_flips, i) * (0.5) ** n_flips for i in range(k, n_flips + 1))
    return probability


def flip():
    return random.randint(0, 1)


def do_flips(threshold=0.60, cheat=False):
    # Flip the coin until 2/3 of the flips are heads
    # Abort after 100000 flips and output the average heads
    # and tails
    flips = 0
    heads = 0
    while True:
        flip_result = flip()
        flips += 1
        if flip_result == 0:
            heads += 1
        if flips >= 100:
            break
        if cheat and heads / flips >= threshold and flips >= 10:
            break
    return flips, heads, (heads / flips)


def multiple_tries(runs=100, threshold=0.60):
    # Run flip_until_mostly_heads 100 times and track how often we get 2/3 heads
    averages = []
    for i in range(runs):
        flips, heads, average_heads = do_flips(threshold=threshold)
        print(f"Try {i + 1}: Flips: {flips}, Heads: {heads}, Average Heads: {average_heads}")
        averages.append(average_heads)
    # Print number of times we saw 2/3 heads or better, i.e. count values > 0.6666666666666666 in averages
    count = 0
    for average in averages:
        if average >= threshold:
            count += 1
    print(f"Count of averages > {threshold}: {count} out of {runs}")
    return count/runs


def main():
    threshold = 0.66
    prob = probability_at_least_x_percent_heads(threshold, 100)
    outcome = multiple_tries(100, threshold=threshold)
    print(f"Probability of getting at least {threshold * 100}% heads in 10 flips: {prob}")
    print(f"Actual outcome of multiple tries: {outcome}")


if __name__ == "__main__":
    main()

