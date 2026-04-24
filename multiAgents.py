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

        # Never choose immediate losing states when alternatives exist.
        if successorGameState.isLose():
            return float('-inf')

        # Base score captures food, capsule, win/lose and time-penalty effects.
        score = successorGameState.getScore()

        foodList = newFood.asList()
        currentFood = currentGameState.getFood()

        # Reward immediate food consumption.
        if currentFood[int(newPos[0])][int(newPos[1])]:
            score += 8

        # Use reciprocal distance so nearer food gets much stronger preference.
        if foodList:
            minFoodDistance = min(manhattanDistance(newPos, foodPos) for foodPos in foodList)
            score += 6.0 / (minFoodDistance + 1.0)

        # Prefer states with fewer remaining pellets.
        score -= 3.0 * len(foodList)

        # Capsules are valuable, especially with nearby active ghosts.
        capsules = successorGameState.getCapsules()
        if capsules:
            minCapsuleDistance = min(manhattanDistance(newPos, capPos) for capPos in capsules)
            score += 2.0 / (minCapsuleDistance + 1.0)

        minActiveGhostDistance = float('inf')
        for ghostState, scaredTime in zip(newGhostStates, newScaredTimes):
            ghostPos = ghostState.getPosition()
            distanceToGhost = manhattanDistance(newPos, ghostPos)

            if scaredTime > 0:
                # Chase scared ghosts while edible.
                score += 10.0 / (distanceToGhost + 1.0)
            else:
                minActiveGhostDistance = min(minActiveGhostDistance, distanceToGhost)
                if distanceToGhost <= 1:
                    score -= 1000
                else:
                    score -= 8.0 / distanceToGhost

        # Extra caution when an active ghost is close.
        if minActiveGhostDistance < 3:
            score -= 10.0 / (minActiveGhostDistance + 0.1)

        # Discourage wasting turns.
        if action == Directions.STOP:
            score -= 15

        return score

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
        numAgents = gameState.getNumAgents()

        def minimax(state, depth, agentIndex):
            # Stop on terminal states or when desired ply depth is reached.
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                # Pacman (max node)
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, minimax(successor, nextDepth, nextAgent))
                return value

            # Ghosts (min nodes)
            value = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = min(value, minimax(successor, nextDepth, nextAgent))
            return value

        bestValue = float('-inf')
        bestActions = []
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = minimax(successor, 0, 1 % numAgents)
            if value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)

        return random.choice(bestActions)

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        numAgents = gameState.getNumAgents()

        def alphabeta(state, depth, agentIndex, alpha, beta):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                    # Prune only on strict inequality per project spec.
                    if value > beta:
                        return value
                    alpha = max(alpha, value)
                return value

            value = float('inf')
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value = min(value, alphabeta(successor, nextDepth, nextAgent, alpha, beta))
                # Prune only on strict inequality per project spec.
                if value < alpha:
                    return value
                beta = min(beta, value)
            return value

        alpha = float('-inf')
        beta = float('inf')
        bestValue = float('-inf')
        bestAction = None

        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = alphabeta(successor, 0, 1 % numAgents, alpha, beta)
            if value > bestValue:
                bestValue = value
                bestAction = action
            alpha = max(alpha, bestValue)

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
        numAgents = gameState.getNumAgents()

        def expectimax(state, depth, agentIndex):
            if state.isWin() or state.isLose() or depth == self.depth:
                return self.evaluationFunction(state)

            legalActions = state.getLegalActions(agentIndex)
            if not legalActions:
                return self.evaluationFunction(state)

            nextAgent = (agentIndex + 1) % numAgents
            nextDepth = depth + 1 if nextAgent == 0 else depth

            if agentIndex == 0:
                value = float('-inf')
                for action in legalActions:
                    successor = state.generateSuccessor(agentIndex, action)
                    value = max(value, expectimax(successor, nextDepth, nextAgent))
                return value

            # Ghosts are modeled as choosing uniformly at random.
            probability = 1.0 / len(legalActions)
            value = 0.0
            for action in legalActions:
                successor = state.generateSuccessor(agentIndex, action)
                value += probability * expectimax(successor, nextDepth, nextAgent)
            return value

        bestValue = float('-inf')
        bestActions = []
        for action in gameState.getLegalActions(0):
            successor = gameState.generateSuccessor(0, action)
            value = expectimax(successor, 0, 1 % numAgents)
            if value > bestValue:
                bestValue = value
                bestActions = [action]
            elif value == bestValue:
                bestActions.append(action)

        return random.choice(bestActions)

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: Linear feature-based state evaluation using:
    - game score as baseline
    - reciprocal distance to nearest food and capsule
    - penalties for remaining food/capsules
    - strong penalties near active ghosts
    - rewards for approaching scared ghosts
    """
    if currentGameState.isWin():
        return float('inf')
    if currentGameState.isLose():
        return float('-inf')

    pos = currentGameState.getPacmanPosition()
    foodList = currentGameState.getFood().asList()
    capsuleList = currentGameState.getCapsules()
    ghostStates = currentGameState.getGhostStates()

    score = currentGameState.getScore()

    # Food pressure: clear remaining pellets while favoring nearby food.
    if foodList:
        nearestFoodDist = min(manhattanDistance(pos, foodPos) for foodPos in foodList)
        score += 12.0 / (nearestFoodDist + 1.0)
    score -= 4.0 * len(foodList)

    # Capsules: encourage collecting them, especially when still available.
    if capsuleList:
        nearestCapsuleDist = min(manhattanDistance(pos, capPos) for capPos in capsuleList)
        score += 4.0 / (nearestCapsuleDist + 1.0)
    score -= 15.0 * len(capsuleList)

    # Ghost interaction: avoid active ghosts, chase scared ghosts.
    activeGhostDists = []
    for ghostState in ghostStates:
        ghostPos = ghostState.getPosition()
        ghostDist = manhattanDistance(pos, ghostPos)

        if ghostState.scaredTimer > 0:
            score += 18.0 / (ghostDist + 1.0)
        else:
            activeGhostDists.append(ghostDist)
            if ghostDist <= 1:
                score -= 500
            else:
                score -= 6.0 / ghostDist

    # Extra safety term against the closest active ghost.
    if activeGhostDists:
        closestActive = min(activeGhostDists)
        score -= 14.0 / (closestActive + 0.2)

    return score

# Abbreviation
better = betterEvaluationFunction
