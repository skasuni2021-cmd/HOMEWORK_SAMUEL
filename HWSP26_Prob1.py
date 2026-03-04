# region imports
#JES MISSING CODE

# endregion

# region function declarations
def main():
    """
    This function rolls a fair die 1000 times and tallies the number of 1's, 2's etc.  Finally,
    it calculates and outputs the probability of each possible score.
    :return: nothing
    """
    scores = #JES MISSING CODE  # create a list with 6 elements/items initialized to 0's
    n = #JES MISSING CODE  # how many times to roll the die
    for i in range(n):  # each time through the loop, roll die and increment a score
        score = Die.rollFairDie()  # score = 1 to 6
        scores[#JES MISSING CODE] += 1  # increment score-1 item b/c 0 indexing start
    # print the result
    #JES MISSING CODE


def main2():
    """
    This function rolls a fair die 10000 times and tallys the number of 1's, 2's etc.  Finally,
    it calculates and outputs the probability of each possible score.
    :return: nothing
    """
    #JES MISSING CODE
    pass


def main3():
    """
    This function rolls an unfair die 10000 times and tallys the number of 1's, 2's etc.  Finally,
    it calculates and outputs the probability of each possible score.
    :return: nothing
    """
    #JES MISSING CODE
    pass


# endregion

# this if statement prevents these calls if this file is imported as a module.
if __name__ == "__main__":
    main()
    main2()
    main3()
