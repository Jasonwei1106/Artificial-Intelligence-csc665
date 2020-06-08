# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """
    # constructor to make a list that shows visited path
    def __init__(self):
        self.visited = []


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"
        #first in first out
        self.visited.insert(0,gameState.generatePacmanSuccessor(legalMoves[chosenIndex]).getPacmanPosition())
        if len(self.visited) > 5:
            self.visited.pop()
        # print(gameState.generatePacmanSuccessor(legalMoves[chosenIndex]).getPacmanPosition())

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        "*** YOUR CODE HERE ***"
        closeFood = 0
        # print("1",newFood)
        #if there is food close by then it's a good position
        for row in newFood[newPos[0] - 2:newPos[0] + 2]:
            for col in row[newPos[1] - 2:newPos[1] + 2]:
                if col:
                    closeFood += 1
        eatGhost = 0
        #iterate among all ghosts
        for index in range(len(newGhostStates)):
            dist = util.manhattanDistance(newPos, newGhostStates[index].getPosition())  #calculate the distance between new position and newghost position
            if dist == 0:   #if it is ate gohost it is a lose
                eatGhost = -10000
            elif dist <= newScaredTimes[index] / 1.5:  # just in case the randomness that's why I do 1.5
                eatGhost += (1. / dist) * 100    # if the ghost is scared and the more close it is the better it will be
            elif dist <= 3:         #3 is a good number because even if paceman and ghost walk face to face it will still have onedistance
                eatGhost -= (1. / dist) * 100  #the more distance paceman have the safter paceman will be

        food = 0
        if newPos in currentGameState.getFood().asList():    #if the new position is on the food, then points will be higher
            food = 10

        visit = 0
        if newPos in self.visited:  #new position shouldn't be just running cycle
            visit = -100

        return closeFood + eatGhost + visit + food

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        value = self.value(gameState,0, 0)
        return value[0]

    def value(self, gameState, depth, agent):
        if agent >= gameState.getNumAgents():   #if we go through all the agents then we reset agent to zero and depth plus one
            agent = 0
            depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():   # is the game end then return score.
            return (None, self.evaluationFunction(gameState))
        if agent == 0:
            # if agent is zero then it's pacman's turn
            return self.max_value(gameState, depth, agent)   #call max function
        else:
            # if agent is nozero then it's ghosts' turn
            return self.min_value(gameState, depth, agent)   #call min function


    def min_value(self, gameState, depth,agent):
        actions = gameState.getLegalActions(agent)    #get the actions when each time they loop through successor
        if len(actions) == 0:    #if no action then return the score.
            return (None, self.evaluationFunction(gameState))

        min_val = (None, float("inf"))   # init the minvalue as positive infinity
        for action in actions:
            state = gameState.generateSuccessor(agent, action)   #get the state for ghost successor
            value = self.value(state, depth, agent+1)   #in order to make recursive works, we need go to the next agent
            if value[1] < min_val[1]:
                min_val = (action, value[1])   #get the min_val and the action accordingly

        return min_val

    # the max function is the same as min function, the only difference is that the agent is 0
    def max_value(self, gameState, depth,agent):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return (None, self.evaluationFunction(gameState))

        max_val = (None, -float("inf"))  # init the maxvalue as negative infinity
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            value = self.value(state, depth,agent + 1)
            if value[1] > max_val[1]:
                max_val = (action, value[1])

        return max_val


#it is simliar to minmax function and I just write it based on the slides.
class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value = self.value(gameState, 0, 0, -float("inf"), float("inf"))
        return value[0]

    def value(self, gameState, depth, agent,alpha,beta):
        if agent >= gameState.getNumAgents():  # if we go through all the agents then we reset agent to zero and depth plus one
            agent = 0
            depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():  # is the game end then return score.
            return (None, self.evaluationFunction(gameState))
        if agent == 0:
            # if agent is zero then it's pacman's turn
            return self.max_value(gameState, depth, agent, alpha, beta)  # call max function
        else:
            # if agent is nozero then it's ghosts' turn
            return self.min_value(gameState, depth, agent, alpha, beta)  # call min function

    def min_value(self, gameState, depth, agent, alpha, beta):
        actions = gameState.getLegalActions(agent)  # get the actions when each time they loop through successor
        if len(actions) == 0:  # if no action then return the score.
            return (None, self.evaluationFunction(gameState))

        min_val = (None, float("inf"))  # init the minvalue as positive infinity
        for action in actions:
            state = gameState.generateSuccessor(agent, action)  # get the state for ghost successor
            value = self.value(state, depth,agent + 1,alpha,beta)  # in order to make recursive works, we need go to the next agent
            if value[1] < min_val[1]:
                min_val = (action, value[1])  # get the max_val and the action accordingly
            if min_val[1] < alpha:
                return min_val
            beta = min(beta,min_val[1])

        return min_val

    def max_value(self, gameState, depth, agent, alpha, beta):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return (None, self.evaluationFunction(gameState))

        max_val = (None, -float("inf"))   # init the minvalue as negative infinity
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            value = self.value(state, depth, agent + 1,alpha,beta)
            if value[1] > max_val[1]:
                max_val = (action, value[1])
            if max_val[1] > beta:
                return max_val
            alpha = max(alpha, max_val[1])

        return max_val


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        value = self.value(gameState,0, 0)
        return value[0]

    def value(self, gameState, depth, agent):
        if agent >= gameState.getNumAgents():   #if we go through all the agents then we reset agent to zero and depth plus one
            agent = 0
            depth += 1
        if depth == self.depth or gameState.isWin() or gameState.isLose():   # is the game end then return score.
            return (None, self.evaluationFunction(gameState))
        if agent == 0:
            # if agent is zero then it's pacman's turn
            return self.max_value(gameState, depth, agent)   #call max function
        else:
            # if agent is nozero then it's ghosts' turn
            return self.min_value(gameState, depth, agent)   #call min function


    def min_value(self, gameState, depth,agent):
        actions = gameState.getLegalActions(agent)    #get the actions when each time they loop through successor
        if len(actions) == 0:    #if no action then return the score.
            return (None, self.evaluationFunction(gameState))
        probablity = 1/len(actions)   #each movements have the probablitly of 1/ number of actions
        val = 0
        for action in actions:
            state = gameState.generateSuccessor(agent, action)   #get the state for ghost successor
            value = self.value(state, depth, agent+1)   #in order to make recursive works, we need go to the next agent
            val += value[1] * probablity
        return (None,val)

    # the max function is the same as min function, the only difference is that the agent is 0
    def max_value(self, gameState, depth,agent):
        actions = gameState.getLegalActions(0)
        if len(actions) == 0:
            return (None, self.evaluationFunction(gameState))

        max_val = (None, -float("inf"))
        for action in actions:
            state = gameState.generateSuccessor(0, action)
            value = self.value(state, depth,agent + 1)
            if value[1] > max_val[1]:
                max_val = (action, value[1])

        return max_val

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: the state should be evalutaed based on the food and ghosts. the thing which will change is if there is the wall
    in between then the paceman is safe so the evalutation should be increase
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    eatGhost = False
    if newScaredTimes[0] > 5:  #if the ghost is scared in 5 moves then change eatGhost to true.
        eatGhost = True
    food_dist = []   # create an empty to store all the distance from the new pos to food
    food_min_dist = 0
    food_eva = 0
    food_list = currentGameState.getFood().asList()
    for food in food_list:    #loop through all the food list and add all the distances from new pos to foods
        food_dist.append(manhattanDistance(newPos, food))
    if len(food_dist) > 0:
        food_min_dist = min(food_dist)    #get the minium food distance from the list
    if food_min_dist > 0:
        food_eva = (2/food_min_dist)
    else:
        food_eva = 0
    if eatGhost:
        food_eva= food_eva* 10    # if the ghost is scared then the evaluation should be higher

    ghost_dist = []   #make an empty list to put the ghost distance
    ghost_min_dist = 0
    ghost_eva = 0
    for ghost in newGhostStates:
        ghost_dist.append(manhattanDistance(newPos,ghost.getPosition()))    #loop through all the ghosts and add all the distances from new post to ghosts
    if len(ghost_dist) > 0:
        ghost_min_dist = min(ghost_dist)     #get the minium Ghost distance from the list
    if ghost_min_dist >0:
        ghost_eva = -2/ghost_min_dist

    # Ghost are scared, pacman should be atracted by them
    if eatGhost:
        ghost_eva = ghost_eva * -10

    score = currentGameState.getScore()

    return score + ghost_eva + food_eva

# Abbreviation
better = betterEvaluationFunction
