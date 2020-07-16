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
import numpy as np
import copy

from game import Agent

def closestDot(originPoint, otherPoints):
    """
    It returns the closest point from a origin to one point of a given collection
    """
    closestCost = 9999999

    for point in otherPoints:
        cost = util.manhattanDistance(originPoint, point)
        if cost < closestCost:
            closestCost = cost
            closestPoint = point

    if closestCost == 9999999:
        return None
    else:
        return (closestCost, closestPoint)

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

        "*** YOUR CODE HERE ***"
        return successorGameState.getScore()


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
        self.maxScore = 0 #Se incicializa en 0

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
        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()

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
        util.raiseNotDefined()

class MCT_Node:
    """Node in the Monte Carlo search tree, keeps track of the children states."""

    def __init__(self, parent=None, state=None, U=0, N=0):
        self.__dict__.update(parent=parent, state=state, U=U, N=N)
        self.children = {}
        self.actions = None

class MonteCarloAgent(MultiAgentSearchAgent):

    def ucb(self, n, C = 1.4):
        if n.N == 0:
            return np.inf    
        else:
            return (n.U / n.N) + C * np.sqrt(np.log(n.parent.N) / n.N)

    def select(self, n):
        """select a leaf node in the tree"""
        if n.children:
            return self.select(max(n.children.keys(), key=self.ucb))
        else:
            return n

    def expand(self, n, k):
        """expand the leaf node by adding all its children states"""
        legal_actions = n.state.getLegalActions()

        if not n.children and not n.state.isWin():
            n.children = {MCT_Node(state=n.state.generateSuccessor(0, action), parent=n):
             action for action in random.sample(legal_actions, k = min(k,len(legal_actions)))}
        return self.select(n)

    def simulate(self, gameState):
        """simulate the utility of current state by random picking a step"""
        state = copy.deepcopy(gameState)
        numeroIteraciones = 30
        iteracion = 1
        while (not state.isWin()) and (not state.isLose()) and (iteracion <= numeroIteraciones):
            legalActions = state.getLegalActions()
            action = random.choice(list(legalActions))
            state = state.generateSuccessor(0, action)
            iteracion += 1
        v = self.evaluationFunction(state)
        return -v

    def backprop(self, n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        n.N += 1
        if n.parent:
            self.backprop(n.parent, -utility)

    def monte_carlo_tree_search(self, gameState, N = 20, k = 3):

        root = MCT_Node(state=gameState)

        for _ in range(N):
            leaf = self.select(root)
            child = self.expand(leaf, k)
            result = self.simulate(child.state)
            self.backprop(child, result)

        max_state = max(root.children, key=lambda p: p.N)

        return root.children.get(max_state)

    def getAction(self, gameState):
        return self.monte_carlo_tree_search(gameState)

class MonteCarloAgentModified(MultiAgentSearchAgent):

    def mixMax(self, n, C = 1.4):
        if n.N == 0:
            return np.inf    
        else:
            Q = 0.1
            return Q*self.maxScore + (1-Q)*(n.U / n.N) +  C * np.sqrt(np.log(n.parent.N) / n.N)

    """
    def ucb(self, n, C = 1.4):
        if n.N == 0:
            return np.inf    
        else:
            return (n.U / n.N) + C * np.sqrt(np.log(n.parent.N) / n.N)
    """

    def select(self, n):
        """select a leaf node in the tree"""
        if n.children:
            #return self.select(max(n.children.keys(), key=self.ucb))
            return self.select(max(n.children.keys(), key=self.mixMax))
        else:
            return n

    def expand(self, n, k):
        """expand the leaf node by adding all its children states"""
        #k = 3 # Maximo numero de nodos a colocar en el arbol en la expansion
        legal_actions = n.state.getLegalActions()
        
        if not n.children and not n.state.isWin():
            n.children = {MCT_Node(state=n.state.generateSuccessor(0, action), parent=n):
             action for action in random.sample(legal_actions, k = min(k,len(legal_actions)))}
        return self.select(n)

    def simulate(self, gameState):
        """simulate the utility of current state by random picking a step"""
        global bestScore
        state = copy.deepcopy(gameState)
        numeroIteraciones = 20
        iteracion = 1
        while (not state.isWin()) and (not state.isLose()) and (iteracion <= numeroIteraciones):
            
            legalActions = state.getLegalActions()
            # Aplicaremos una heurística a fin de que la acción a tomar no sea totalmente aleatoria
            # Choose one of the best actions
            scores = [self.evaluationFunction_heuristic(state, action) for action in legalActions]
            bestScore = max(scores)
            bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
            # Pick randomly among the best
            action = legalActions[random.choice(bestIndices)]
            #action = random.choice(list(legalActions))
            state = state.generateSuccessor(0, action)
            iteracion += 1
        return -bestScore

    def backprop(self, n, utility):
        """passing the utility back to all parent nodes"""
        if utility > 0:
            n.U += utility
        n.N += 1
        if n.parent:
            self.backprop(n.parent, -utility)

    def monte_carlo_tree_search(self, gameState, N = 3, k = 5):

        root = MCT_Node(state=gameState)

        for _ in range(N):
            leaf = self.select(root)
            child = self.expand(leaf, k)
            result = self.simulate(child.state)
            self.backprop(child, result)

        max_state = max(root.children, key=lambda p: p.N)

        return root.children.get(max_state)

    def evaluationFunction_heuristic(self, currentGameState, action):

        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        
        score = 0
        phantom_distances = [util.manhattanDistance(
            newPos, ghost_position) for ghost_position in successorGameState.getGhostPositions()]

        closest_food = closestDot(newPos, newFood.asList())
        
        if closest_food is not None:
            score += (-int(closest_food[0]) * 50) - (len(newFood.asList())* 1000)
        if action == 'stop':
            score -= 10000

        closest_phantom = min(phantom_distances)
        if closest_phantom <= 2 and newScaredTimes[phantom_distances.index(closest_phantom)] < closest_phantom:
            score = -float('inf')
        if newScaredTimes[phantom_distances.index(closest_phantom)] > closest_phantom:
            score += 100
        
        #print("ACTION =>",action, "SCORE=>", score, "--- [ph: ",closest_phantom," cf:",closest_food)
        return score

    def getAction(self, gameState):
        return self.monte_carlo_tree_search(gameState)