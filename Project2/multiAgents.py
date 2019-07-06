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
    previousLoc = ()


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
        self.previousLoc = gameState.getPacmanPosition()
        return legalMoves[chosenIndex]

    def getClosestFood(self, foodPellets, pos):
        closestFood = 1000000
        closestFoodPos = ()
        for food in foodPellets:
            if food != pos:
                foodDis = manhattanDistance(pos, food)
                if foodDis < closestFood:
                    closestFood = foodDis
                    closestFoodPos = food
        return (closestFood, closestFoodPos)


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
        foodPellets = newFood.asList()
        currPos = currentGameState.getPacmanPosition()
        closestFood, closestFoodPos = self.getClosestFood(foodPellets, newPos)  
        closestGhostDist = 1000000
        closestGhost = ()
        for ghost in newGhostStates:
            distToGhost = manhattanDistance(newPos, ghost.getPosition())
            if distToGhost < closestGhostDist:
                closestGhostDist = distToGhost
                closestGhost = ghost

        if closestGhostDist > 0:
            if closestGhost.scaredTimer / closestGhostDist > 1:
                return successorGameState.getScore() + 25 / closestGhostDist + 1 / closestFood

            if closestGhostDist <= 2:
                return -len(foodPellets) - 5 / closestGhostDist + 1 / closestFood

        if closestFoodPos != ():
            if newPos == self.previousLoc: 
                return -len(foodPellets) - 1 + 1 / closestFood

        if currPos == newPos:
            return -len(foodPellets) - .5 + 1 / closestFood

        return -len(foodPellets) + 1 / closestFood
      

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
        def minValue(gameState, agent, depth):
            minVal = float('inf')
            successors = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
            if agent == gameState.getNumAgents() - 1:
                depth += 1
                agent = 0
            else: 
                agent += 1
            for successor in successors:
                minVal = min(minVal, value(successor, agent, depth))
            return minVal 

        def maxValue(gameState, agent, depth):
            maxVal = -float('inf')
            successors = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
            agent = 1
            for successor in successors:
                maxVal = max(maxVal, value(successor, agent, depth))
            return maxVal

        def value(gameState, agent, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, agent, depth)
            else:
                return minValue(gameState, agent, depth)

        legalMoves = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [value(successor, 1, 0) for successor in successors]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]
        return legalMoves[chosenIndex]

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minValue(gameState, agent, depth, alpha, beta):
            minVal = float('inf')
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                if agent == gameState.getNumAgents() - 1:
                    minVal = min(minVal, value(successor, 0, depth + 1, alpha, beta))
                else: 
                    minVal = min(minVal, value(successor, agent + 1, depth, alpha, beta)) 
                if minVal < alpha:
                    return minVal
                beta = min(beta, minVal)
            return minVal 

        def maxValue(gameState, agent, depth, alpha, beta):
            maxVal = -float('inf')
            for action in gameState.getLegalActions(agent):
                successor = gameState.generateSuccessor(agent, action)
                maxVal = max(maxVal, value(successor, 1, depth, alpha, beta))
                if maxVal > beta:
                    return maxVal
                alpha = max(alpha, maxVal)
            return maxVal

        def value(gameState, agent, depth, alpha, beta):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, agent, depth, alpha, beta)
            else:
                return minValue(gameState, agent, depth, alpha, beta)

        legalMoves = gameState.getLegalActions(0)
        alpha = -float('inf')
        beta = float('inf')
        bestScore = -float('inf')
        bestAction = legalMoves[0]
        for action in legalMoves:
            successor = gameState.generateSuccessor(0, action)
            score = value(successor, 1, 0, alpha, beta)
            alpha = max(alpha, score)
            if score > bestScore:
                bestScore = score
                bestAction = action
        return bestAction

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
        "*** YOUR CODE HERE ***"
        def expValue(gameState, agent, depth):
            expVal = 0
            successors = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
            if agent == gameState.getNumAgents() - 1:
                depth += 1
                agent = 0
            else: 
                agent += 1
            p = 1.0 / len(successors)
            for successor in successors:
                expVal += p * value(successor, agent, depth)
            return expVal

        def maxValue(gameState, agent, depth):
            maxVal = -float('inf')
            successors = [gameState.generateSuccessor(agent, action) for action in gameState.getLegalActions(agent)]
            agent = 1
            for successor in successors:
                maxVal = max(maxVal, value(successor, agent, depth))
            return maxVal

        def value(gameState, agent, depth):
            if gameState.isWin() or gameState.isLose() or depth == self.depth:
                return self.evaluationFunction(gameState)
            if agent == 0:
                return maxValue(gameState, agent, depth)
            else:
                return expValue(gameState, agent, depth)

        legalMoves = gameState.getLegalActions(0)
        successors = [gameState.generateSuccessor(0, action) for action in legalMoves]
        scores = [value(successor, 1, 0) for successor in successors]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = bestIndices[0]
        return legalMoves[chosenIndex]

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: In this function we kept track of the closest ghost, closest pellets and distance to 
    all pellets, distances to capsules, and the scared time of the closest ghost. If the closest ghost 
    was scared we prioritized eating it because of how many extra points it gives, and if the closest ghost 
    wasn't scared we prioritized eating pellets. If the closest ghost wasn't scared and got too close we 
    prioritized getting away from it. 
    """
    "*** YOUR CODE HERE ***"
    def getClosest(locs, pos):
        closest = 100000
        closestPos = ()
        totalDist = 0
        for loc in locs:
            dist = manhattanDistance(pos, loc)
            totalDist += dist
            if dist < closest:
                closest = dist
                closestPos = loc
        return (closest, closestPos, totalDist)

    #gamestate stuff
    currPos = currentGameState.getPacmanPosition()
    Food = currentGameState.getFood()
    GhostStates = currentGameState.getGhostStates()
    ScaredTimes = [ghostState.scaredTimer for ghostState in GhostStates]
    Capsules = currentGameState.getCapsules()
    value = currentGameState.getScore()

    #find out stuff about the pellets remaining
    foodPellets = Food.asList()
    numPellets = len(foodPellets)
    closestFood, closestFoodPos, totalFoodDist = getClosest(foodPellets, currPos)

    #find closest ghost
    closestGhostDist = 1000000
    closestGhost = ()
    for ghost in GhostStates:
        distToGhost = manhattanDistance(currPos, ghost.getPosition())
        if distToGhost < closestGhostDist:
            closestGhostDist = distToGhost
            closestGhost = ghost

    #find capsules to make ghosts scared, looking at pacman actions I'm not sure if this works or only the 
    #scared timer part decides if a capsule gets eaten
    closestCapsuleDist, closestCapsule, totalCapsuleDist = getClosest(Capsules, currPos)
    if currPos in Capsules:
        value += 25
    if closestCapsuleDist == 1:
        value += 1

    
    #if ghost is scared prioritize eating them
    if closestGhost.scaredTimer != 0:
        if closestGhostDist > 0:
            value += closestGhost.scaredTimer / closestGhostDist
        if closestGhostDist == 0:
            value += 1000

    #avoid getting caught by ghosts
    if closestGhostDist == 1 and closestGhost.scaredTimer == 0:
        value -= 100

    #incentivize eating pellets if ghosts aren't scared and avoid getting stuck trying to choose
    if totalFoodDist > 0 and closestGhost.scaredTimer == 0:
        value -= totalFoodDist / numPellets
    value += 1 / closestFood

    return value

# Abbreviation
better = betterEvaluationFunction
