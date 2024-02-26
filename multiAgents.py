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
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
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
        bestIndices = [
            index for index in range(len(scores)) if scores[index] == bestScore
        ]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
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
        # print("successorGameState", successorGameState)
        # print("newPos", newPos)
        # print("newFood", newFood.asList())
        # for i in newGhostStates:
        #     print(i)
        # # print(newGhostStates[0])
        # print("newScaredTimes", newScaredTimes)
        "*** YOUR CODE HERE ***"
        score = successorGameState.getScore()

        # calcualate ghost distances
        for ghostState in newGhostStates:
            ghostDistance = util.manhattanDistance(ghostState.getPosition(), newPos)
            if ghostDistance < 3:
                score -= 50
            elif ghostDistance < 6:
                score -= 10

        # calculate food distances
        foodDistances = [
            util.manhattanDistance(food, newPos) for food in newFood.asList()
        ]
        if foodDistances:
            closest_food_distance = min(foodDistances)
            score += 10 / closest_food_distance

        return score


def scoreEvaluationFunction(currentGameState: GameState):
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

    def __init__(self, evalFn="scoreEvaluationFunction", depth="2"):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState: GameState):
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
        action = Directions.STOP
        actions = gameState.getLegalActions(0)
        v = float("-inf")
        for action in actions:
            minimax_value = self.minimax(gameState.generateSuccessor(0, action), 0, 1)
            if v < minimax_value:
                res = action
                v = minimax_value
        return res
        # util.raiseNotDefined()

    def minimax(self, gameState, tree_depth, agent_index):
        # terminal state
        if gameState.isLose() or gameState.isWin() or tree_depth == self.depth:
            return self.evaluationFunction(gameState)
        # ghost / min state
        elif agent_index != 0:
            return self.ghost_min_value(gameState, tree_depth, agent_index)
        # pacman / max
        else:
            return self.pacman_max_value(gameState, tree_depth, agent_index=0)

    def ghost_min_value(self, gameState, tree_depth, agent_index):
        v = float("inf")
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            # check if previous state agent or not
            if agent_index != (gameState.getNumAgents() - 1):
                v = min(
                    v,
                    self.minimax(
                        successor,
                        tree_depth,
                        agent_index + 1,
                    ),
                )
            else:
                v = min(
                    v,
                    self.minimax(
                        successor,
                        tree_depth + 1,
                        0,
                    ),
                )
        return v

    def pacman_max_value(self, gameState, tree_depth, agent_index=0):
        v = float("-inf")
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            v = max(
                v,
                self.minimax(successor, tree_depth, 1),
            )
        return v


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        action = Directions.STOP
        actions = gameState.getLegalActions(0)
        a = float("-inf")
        b = float("inf")
        v = float("-inf")
        for action in actions:
            minimax_value = max(
                self.minimax(gameState.generateSuccessor(0, action), 0, a, b, 1), v
            )
            if a < minimax_value:
                res = action
                a = minimax_value
        return res

    def minimax(self, gameState, tree_depth, a, b, agent_index):
        if gameState.isLose() or gameState.isWin() or tree_depth == self.depth:
            return self.evaluationFunction(gameState)
        elif agent_index != 0:
            return self.ghost_min_value(gameState, tree_depth, a, b, agent_index)
        else:
            return self.pacman_max_value(gameState, tree_depth, a, b, agent_index=0)

    def ghost_min_value(self, gameState, tree_depth, a, b, agent_index):
        v = float("inf")
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            if agent_index != (gameState.getNumAgents() - 1):
                v = min(
                    v,
                    self.minimax(
                        successor,
                        tree_depth,
                        a,
                        b,
                        agent_index + 1,
                    ),
                )
            else:
                v = min(
                    v,
                    self.minimax(
                        successor,
                        tree_depth + 1,
                        a,
                        b,
                        0,
                    ),
                )
            # pruning if applicable
            if v < a:
                return v
            else:
                b = min(b, v)
        return v

    def pacman_max_value(self, gameState, tree_depth, a, b, agent_index=0):
        v = float("-inf")
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            v = max(
                v,
                self.minimax(successor, tree_depth, a, b, 1),
            )
            if b < v:  # Prune if necessary
                return v
            else:
                a = max(a, v)
        return v


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        # util.raiseNotDefined()
        action = Directions.STOP
        actions = gameState.getLegalActions(0)
        v = float("-inf")
        for action in actions:
            minimax_value = self.expectimax(
                gameState.generateSuccessor(0, action), 0, 1
            )
            if v < minimax_value:
                res = action
                v = minimax_value
        return res
        # util.raiseNotDefined()

    def expectimax(self, gameState, tree_depth, agent_index):
        if gameState.isLose() or gameState.isWin() or tree_depth == self.depth:
            return self.evaluationFunction(gameState)
        elif agent_index != 0:
            return self.ghost_exp_value(gameState, tree_depth, agent_index)
        else:
            return self.pacman_max_value(gameState, tree_depth, agent_index=0)

    def ghost_exp_value(self, gameState, tree_depth, agent_index):
        v = 0  # want to be optimistic and not start with inf
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            # probability based on assumption that adversary chooses getLegalActions uniformly at random
            p = 1.0 / len(actions)
            if agent_index != (
                gameState.getNumAgents() - 1
            ):  # if not previous state agent
                v += p * self.expectimax(successor, tree_depth, agent_index + 1)
            else:
                v += p * self.expectimax(successor, tree_depth + 1, 0)
        return v

    def pacman_max_value(self, gameState, tree_depth, agent_index=0):
        v = float("-inf")
        actions = gameState.getLegalActions(agent_index)
        for action in actions:
            successor = gameState.generateSuccessor(agent_index, action)
            v = max(
                v,
                self.expectimax(successor, tree_depth, 1),
            )
        return v


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <
        this evalFn takes into factor: calculated distances between pacman and nearest food pellets
        and also the nearest ghosts. It also takes into consideration the ghost's scared time for
        pacman agent to prioritize or not priortize eating or to avoid them.
        All these are taken into consideration when assigning a score to states.
    >
    """
    "*** YOUR CODE HERE ***"
    # util.raiseNotDefined()
    # current game state infos
    pacmanPosition = currentGameState.getPacmanPosition()
    foodGrid = currentGameState.getFood()
    ghostStates = currentGameState.getGhostStates()
    scaredTimes = [ghostState.scaredTimer for ghostState in ghostStates]
    closestFoodDistance = float("inf")
    closestGhostDistance = float("inf")
    totalScaredTime = sum(scaredTimes)
    score = currentGameState.getScore()

    # calculate distance to the nearest food
    for food in foodGrid.asList():
        distanceToFood = util.manhattanDistance(pacmanPosition, food)
        closestFoodDistance = min(closestFoodDistance, distanceToFood)

    # calculate distance to thenearest ghost
    for ghostState in ghostStates:
        ghostPosition = ghostState.getPosition()
        distanceToGhost = util.manhattanDistance(pacmanPosition, ghostPosition)
        closestGhostDistance = min(closestGhostDistance, distanceToGhost)

    # adjust score based on ghost scared or not
    if closestGhostDistance <= 1:
        if totalScaredTime == 0:
            score -= 1000  # avoid not scared ghosts
        else:
            score += 500  # chase scared ghosts if any
    else:
        score += 10 / (closestFoodDistance + 1)  # increase score for closer food

    return score


# Abbreviation
better = betterEvaluationFunction
