def getTrailingZero(hash_value):
    reversed_binary = hash_value[::-1]
    # Find the position of the first '0' from the right
    position = reversed_binary.find('1')
    
    return position


def flajoletMartinAlgo(a, b, x):
    hash = (a * x + b) % 32

    return getTrailingZero(format(hash, "05b"))


if __name__ == "__main__":
    input_stream = [3, 1, 4, 1, 5, 9, 2, 6, 5]
    maxR = 0
    
    for element in input_stream:
        trailing_zero = flajoletMartinAlgo(2, 1, element)
        maxR = max(maxR, trailing_zero)
        
    print(f"a) Number of distinct elements: {2 ** maxR}")

    maxR = 0
    for element in input_stream:
        trailing_zero = flajoletMartinAlgo(3, 7, element)
        maxR = max(maxR, trailing_zero)

    print(f"b) Number of distinct elements: {2 ** maxR}")

    maxR = 0
    for element in input_stream:
        trailing_zero = flajoletMartinAlgo(4, 0, element)
        maxR = max(maxR, trailing_zero)

    print(f"c) Number of distinct elements: {2 ** maxR}")

"""
A problem with this type of hash function is, that it is linear and therefore a hash collision might be found by using multiples of the constants.
My advice would be to use numbers for a and b that have no common factor with 2^k and with each other. This leads to hash values that are more distinct. 
"""