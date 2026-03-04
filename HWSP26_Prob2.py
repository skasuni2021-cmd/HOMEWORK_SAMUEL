#region imports
# $JES MISSING CODE
#endregion

#region functions
def main():  # main function to roll nDice fair dice nRolls times and output probabilities
    """
    This function rolls nDice nRolls times and calculates the probabilities for
    each possible score based on P(7)=nTally/nRolls, where nTally is number times I roll a 7, for example.
    :return: nothing
    """
    nDice = # $JES MISSING CODE  # number of dice
    nMinScore = # $JES MISSING CODE  # total score if each die scores 1
    nMaxScore = # $JES MISSING CODE  # total score if each die scores 6
    nNumScores = # $JES MISSING CODE  # number of possible scores
    nTally = # $JES MISSING CODE  # create a list with (nMaxScore-nMinScore+1) elements/items
    nRolls = # $JES MISSING CODE  # how many times to roll the dice
    for i in range(nRolls):  # each loop rolls dice and increments a score
        # $JES MISSING CODE  # call with N=nDice
        # $JES MISSING CODE  # increment score-nMinScore item b/c 0 indexing start
    print(# $JES MISSING CODE)
    for i in range(nNumScores):  # print the fraction of rolls for each score
        print(# $JES MISSING CODE)

def main2():  # main function to roll nDice unfair dice nRolls times and output probabilities
    """
    This function rolls nDice nRolls times and calculates the probabilities for
    each possible score based on P(7)=nTally/nRolls, where nTally is number times I roll a 7, for example.
    :return: nothing
    """
    nDice = # $JES MISSING CODE  # number of dice
    nMinScore = # $JES MISSING CODE  # total score if each die scores 1
    nMaxScore = # $JES MISSING CODE  # total score if each die scores 6
    nNumScores = # $JES MISSING CODE  # number of possible scores
    nTally = # $JES MISSING CODE  # create a list with (nMaxScore-nMinScore+1) elements/items
    nRolls = # $JES MISSING CODE  # how many times to roll the dice
    for i in range(nRolls):  # each loop rolls dice and increments a score
        # $JES MISSING CODE # call with N=nDice
        # $JES MISSING CODE # increment score-nMinScore item b/c 0 indexing start
    print("After {} rolls of {} dice...".format(nRolls, nDice))
    for i in range(nNumScores):  # print the fraction of rolls for each score
        # $JES MISSING CODE
#endregion

#this if statement prevents these calls if this file is imported as a module.
if __name__ == "__main__":
    main()
    main2()