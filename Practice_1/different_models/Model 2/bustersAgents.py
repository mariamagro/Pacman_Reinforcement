from __future__ import print_function
# bustersAgents.py
# ----------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

from builtins import range
from builtins import object
from wekaI import Weka
import util
from game import Agent
from game import Directions
from keyboardAgents import KeyboardAgent
import inference
import busters
import os


class NullGraphics(object):
    "Placeholder for graphics"

    def initialize(self, state, isBlue=False):
        pass

    def update(self, state):
        pass

    def pause(self):
        pass

    def draw(self, state):
        pass

    def updateDistributions(self, dist):
        pass

    def finish(self):
        pass


class KeyboardInference(inference.InferenceModule):
    """
    Basic inference module for use with the keyboard.
    """

    def initializeUniformly(self, gameState):
        "Begin with a uniform distribution over ghost positions."
        self.beliefs = util.Counter()
        for p in self.legalPositions: self.beliefs[p] = 1.0
        self.beliefs.normalize()

    def observe(self, observation, gameState):
        noisyDistance = observation
        emissionModel = busters.getObservationDistribution(noisyDistance)
        pacmanPosition = gameState.getPacmanPosition()
        allPossible = util.Counter()
        for p in self.legalPositions:
            trueDistance = util.manhattanDistance(p, pacmanPosition)
            if emissionModel[trueDistance] > 0:
                allPossible[p] = 1.0
        allPossible.normalize()
        self.beliefs = allPossible

    def elapseTime(self, gameState):
        pass

    def getBeliefDistribution(self):
        return self.beliefs


class BustersAgent(object):
    "An agent that tracks and displays its beliefs about ghost positions."

    def __init__(self, index=0, inference="ExactInference", ghostAgents=None, observeEnable=True,
                 elapseTimeEnable=True):
        inferenceType = util.lookup(inference, globals())
        self.inferenceModules = [inferenceType(a) for a in ghostAgents]
        self.observeEnable = observeEnable
        self.elapseTimeEnable = elapseTimeEnable
        self.weka = Weka()
        self.weka.start_jvm()
        self.countActions = 0

    def registerInitialState(self, gameState):
        "Initializes beliefs and inference modules"
        import __main__
        self.display = __main__._display
        for inference in self.inferenceModules:
            inference.initialize(gameState)
        self.ghostBeliefs = [inf.getBeliefDistribution() for inf in self.inferenceModules]
        self.firstMove = True

    def observationFunction(self, gameState):
        "Removes the ghost states from the gameState"
        agents = gameState.data.agentStates
        gameState.data.agentStates = [agents[0]] + [None for i in range(1, len(agents))]
        return gameState

    def getAction(self, gameState):
        "Updates beliefs, then chooses an action based on updated beliefs."
        # for index, inf in enumerate(self.inferenceModules):
        #    if not self.firstMove and self.elapseTimeEnable:
        #        inf.elapseTime(gameState)
        #    self.firstMove = False
        #    if self.observeEnable:
        #        inf.observeState(gameState)
        #    self.ghostBeliefs[index] = inf.getBeliefDistribution()
        # self.display.updateDistributions(self.ghostBeliefs)
        return self.chooseAction(gameState)

    def chooseAction(self, gameState):
        "By default, a BustersAgent just stops.  This should be overridden."

        self.countActions += 1

        x = self.getState(gameState)

        move = self.weka.predict("selected_model.model", x, "training_keyboard_new_vars2")

        return move

    def getState(self, gameState):

        if "North" in gameState.getLegalPacmanActions():
            North = 1
        else:
            North = 0

        if "South" in gameState.getLegalPacmanActions():
            South = 1
        else:
            South = 0

        if "East" in gameState.getLegalPacmanActions():
            East = 1
        else:
            East = 0

        if "West" in gameState.getLegalPacmanActions():
            West = 1
        else:
            West = 0

        if gameState.data.ghostDistances[0] == None:
            ghost_distance1 = "?"
        else:
            ghost_distance1 = gameState.data.ghostDistances[0]

        if gameState.data.ghostDistances[1] == None:
            ghost_distance2 = "?"
        else:
            ghost_distance2 = gameState.data.ghostDistances[1]

        if gameState.data.ghostDistances[2] == None:
            ghost_distance3 = "?"
        else:
            ghost_distance3 = gameState.data.ghostDistances[2]

        if gameState.data.ghostDistances[3] == None:
            ghost_distance4 = "?"
        else:
            ghost_distance4 = gameState.data.ghostDistances[3]

        if gameState.getScore() > 620:
            high_score = 1
        else:
            high_score = 0

        info = [self.countActions, gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1], North, South,
                East, West, gameState.getGhostPositions()[0][0], gameState.getGhostPositions()[0][1],
                gameState.getGhostPositions()[1][0], gameState.getGhostPositions()[1][1],
                gameState.getGhostPositions()[2][0], gameState.getGhostPositions()[2][1],
                gameState.getGhostPositions()[3][0], gameState.getGhostPositions()[3][1],
                ghost_distance1, ghost_distance2, ghost_distance3, ghost_distance4, gameState.getScore(), high_score,
                gameState.data.agentStates[0].getDirection()]

        return info


class BustersKeyboardAgent(BustersAgent, KeyboardAgent):
    "An agent controlled by the keyboard that displays beliefs about ghost positions."

    def __init__(self, index=0, inference="KeyboardInference", ghostAgents=None):
        KeyboardAgent.__init__(self, index)
        BustersAgent.__init__(self, index, inference, ghostAgents)
        self.countActions = 0

    def getAction(self, gameState):
        return BustersAgent.getAction(self, gameState)

    def chooseAction(self, gameState):
        self.countActions += 1

        info = self.printLineData(gameState)

        self.file.write(info)

        return KeyboardAgent.getAction(self, gameState)

    def printLineData(self, gameState):

        if not os.path.isfile("dots_kayboard_samemap.arff"):
            self.file = open("dots_kayboard_samemap.arff", "a")
            attributes = ("@relation distance_raw_data\n"
                          "@attribute 'Legal action North' {False, True}\n"
                          "@attribute 'Legal action South' {False, True}\n"
                          "@attribute 'Legal action East' {False, True}\n"
                          "@attribute 'Legal action West' {False, True}\n"
                          "@attribute 'North ghost' numeric\n"
                          "@attribute 'South ghost' numeric\n"
                          "@attribute 'East ghost' numeric\n"
                          "@attribute 'West ghost' numeric\n"
                          "@attribute 'North dot' numeric\n"
                          "@attribute 'South dot' numeric\n"
                          "@attribute 'East dot' numeric\n"
                          "@attribute 'West dot' numeric\n"
                          "@attribute 'Pacman direction' {North, South, East, West, Stop}\n"
                          "@attribute 'Current score' numeric \n"
                          "@attribute 'Future score' numeric \n"
                          "@data\n")

            self.file.write(attributes)

        self.file = open("dots_kayboard_samemap.arff", "a")

        # The information must be a string type, as that is the one that the write method requires
        # For that reason, the concatenation of strings is used with the information we used in the previous method
        # to decide where the pac man should go
        if "North" in gameState.getLegalPacmanActions():
            North = True
        else:
            North = False

        if "South" in gameState.getLegalPacmanActions():
            South = True
        else:
            South = False

        if "East" in gameState.getLegalPacmanActions():
            East = True
        else:
            East = False

        if "West" in gameState.getLegalPacmanActions():
            West = True
        else:
            West = False

        distancer = Distancer(gameState.data.layout)

        min_dist = sys.maxsize

        for i in range(len(gameState.data.ghostDistances)):
            # The ghost is not in jail if the distance is not None
            if gameState.data.ghostDistances[i] != None:
                aux_dist = distancer.getDistance(gameState.getPacmanPosition(), gameState.getGhostPositions()[i])
                if aux_dist < min_dist:
                    min_dist = aux_dist
                    ghost_pos = gameState.getGhostPositions()[i]

        if "North" in gameState.getLegalPacmanActions():
            up = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
            pacmanup = distancer.getDistance(up, ghost_pos)
        else:
            pacmanup = "?"
        if "South" in gameState.getLegalPacmanActions():
            down = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
            pacmandown = distancer.getDistance(down, ghost_pos)
        else:
            pacmandown = "?"
        if "East" in gameState.getLegalPacmanActions():
            right = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
            pacmanright = distancer.getDistance(right, ghost_pos)
        else:
            pacmanright = "?"
        if "West" in gameState.getLegalPacmanActions():
            left = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])
            pacmanleft = distancer.getDistance(left, ghost_pos)

        else:
            pacmanleft = "?"

        # PACDOTS
        pacdots = []
        grid_food = gameState.getFood()
        matrix = []
        transpose = []

        for i in range(grid_food.height):
            actual_line = grid_food[i]

            actual_line = actual_line[::-1]

            matrix.append(actual_line)

        num_rows = len(matrix)
        num_cols = len(matrix[0])

        transpose = [[0 for j in range(num_rows)] for i in range(num_cols)]

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                transpose[j][i] = matrix[i][j]

        for i in range(num_rows):
            for j in range(num_cols):
                if transpose[i][j]:
                    place = (i + 1, j + 1)
                    pacdots.append(place)

        min = sys.maxsize
        short_dot = None
        for i in range(len(pacdots)):
            if distancer.getDistance(gameState.getPacmanPosition(), pacdots[i]) < min:
                min = distancer.getDistance(gameState.getPacmanPosition(), pacdots[i])
                short_dot = pacdots[i]

        if short_dot != None:
            if "North" in gameState.getLegalPacmanActions():
                up = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
                north_dot = distancer.getDistance(up, short_dot)
            else:
                north_dot = "?"
            if "South" in gameState.getLegalPacmanActions():
                down = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
                south_dot = distancer.getDistance(down, short_dot)
            else:
                south_dot = "?"
            if "East" in gameState.getLegalPacmanActions():
                right = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
                right_dot = distancer.getDistance(right, short_dot)
            else:
                right_dot = "?"
            if "West" in gameState.getLegalPacmanActions():
                left = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])
                left_dot = distancer.getDistance(left, short_dot)

            else:
                left_dot = "?"
        else:
            north_dot = "?"
            south_dot = "?"
            right_dot = "?"
            left_dot = "?"

        if self.countActions == 1:
            with open("dots_kayboard_samemap.arff", "r") as arff_file:
                lines = arff_file.readlines()
                if lines[-1] != "@data\n":
                    lines.pop()  # remove the last line

            with open("dots_kayboard_samemap.arff", "w") as arff_file:
                arff_file.write(''.join(lines))

            info = str(North) + ',' + str(South) + ',' + str(East) + ',' + str(West) + ',' + \
                   str(pacmanup) + ',' + str(pacmandown) + ',' + str(pacmanright) + ',' + str(pacmanleft) + ',' + \
                   str(north_dot) + ',' + str(south_dot) + ',' + str(right_dot) + ',' + str(left_dot) + ',' + \
                   str(gameState.data.agentStates[0].getDirection()) + ',' + str(gameState.getScore()) + ','

        else:
            info = str(gameState.getScore()) + '\n' + str(North) + ',' + str(South) + ',' + str(East) + ',' + str(
                West) + ',' + \
                   str(north_dot) + ',' + str(south_dot) + ',' + str(right_dot) + ',' + str(left_dot) + ',' + \
                   str(pacmanup) + ',' + str(pacmandown) + ',' + str(pacmanright) + ',' + str(pacmanleft) + ',' + \
                   str(gameState.data.agentStates[0].getDirection()) + ',' + str(gameState.getScore()) + ','

        # Then the previous string is returned

        return info


from distanceCalculator import Distancer
from game import Actions
from game import Directions
import random, sys

'''Random PacMan Agent'''


class RandomPAgent(BustersAgent):

    def registerInitialState(self, gameState):
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        ##print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def chooseAction(self, gameState):
        move = Directions.STOP
        legal = gameState.getLegalActions(0)  ##Legal position from the pacman
        move_random = random.randint(0, 3)
        if (move_random == 0) and Directions.WEST in legal:  move = Directions.WEST
        if (move_random == 1) and Directions.EAST in legal: move = Directions.EAST
        if (move_random == 2) and Directions.NORTH in legal:   move = Directions.NORTH
        if (move_random == 3) and Directions.SOUTH in legal: move = Directions.SOUTH
        return move


class GreedyBustersAgent(BustersAgent):
    "An agent that charges the closest ghost."

    def registerInitialState(self, gameState):
        "Pre-computes the distance between every two points."
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)

    def chooseAction(self, gameState):
        """
        First computes the most likely position of each ghost that has
        not yet been captured, then chooses an action that brings
        Pacman closer to the closest ghost (according to mazeDistance!).

        To find the mazeDistance between any two positions, use:
          self.distancer.getDistance(pos1, pos2)

        To find the successor position of a position after an action:
          successorPosition = Actions.getSuccessor(position, action)

        livingGhostPositionDistributions, defined below, is a list of
        util.Counter objects equal to the position belief
        distributions for each of the ghosts that are still alive.  It
        is defined based on (these are implementation details about
        which you need not be concerned):

          1) gameState.getLivingGhosts(), a list of booleans, one for each
             agent, indicating whether or not the agent is alive.  Note
             that pacman is always agent 0, so the ghosts are agents 1,
             onwards (just as before).

          2) self.ghostBeliefs, the list of belief distributions for each
             of the ghosts (including ghosts that are not alive).  The
             indices into this list should be 1 less than indices into the
             gameState.getLivingGhosts() list.
        """
        pacmanPosition = gameState.getPacmanPosition()
        legal = [a for a in gameState.getLegalPacmanActions()]
        livingGhosts = gameState.getLivingGhosts()
        livingGhostPositionDistributions = \
            [beliefs for i, beliefs in enumerate(self.ghostBeliefs)
             if livingGhosts[i + 1]]
        return Directions.EAST


class BasicAgentAA(BustersAgent):

    def registerInitialState(self, gameState):

        # In order to open the document just once, so that this does not produce any lag, the document is
        # opened here as an attribute of the initial state of the BasicAgent

        if not os.path.isfile("PacManState.arff"):
            self.file = open("PacManState.arff", "a")
            header = 'Tick, Pacman position, Legal actions, Ghosts positions, Ghosts distances, Movement'
            self.file.write(header)

        self.file = open("PacManState.arff", "a")
        BustersAgent.registerInitialState(self, gameState)
        self.distancer = Distancer(gameState.data.layout, False)
        self.countActions = 0

    ''' Example of counting something'''

    def countFood(self, gameState):
        food = 0
        for width in gameState.data.food:
            for height in width:
                if (height == True):
                    food = food + 1
        return food

    ''' Print the layout'''

    def printGrid(self, gameState):
        table = ""
        # print(gameState.data.layout) ## Print by terminal
        for x in range(gameState.data.layout.width):
            for y in range(gameState.data.layout.height):
                food, walls = gameState.data.food, gameState.data.layout.walls
                table = table + gameState.data._foodWallStr(food[x][y], walls[x][y]) + ","
        table = table[:-1]
        return table

    def printInfo(self, gameState):
        print("---------------- TICK ", self.countActions, " --------------------------")
        # Map size
        width, height = gameState.data.layout.width, gameState.data.layout.height
        print("Width: ", width, " Height: ", height)
        # Pacman position
        print("Pacman position: ", gameState.getPacmanPosition())
        # Legal actions for Pacman in current position
        print("Legal actions: ", gameState.getLegalPacmanActions())
        # Pacman direction
        print("Pacman direction: ", gameState.data.agentStates[0].getDirection())
        # Number of ghosts
        print("Number of ghosts: ", gameState.getNumAgents() - 1)
        # Alive ghosts (index 0 corresponds to Pacman and is always false)
        print("Living ghosts: ", gameState.getLivingGhosts())
        # Ghosts positions
        print("Ghosts positions: ", gameState.getGhostPositions())
        # Ghosts directions
        print("Ghosts directions: ",
              [gameState.getGhostDirections().get(i) for i in range(0, gameState.getNumAgents() - 1)])
        # Manhattan distance to ghosts
        print("Ghosts distances: ", gameState.data.ghostDistances)
        # Pending pac dots
        print("Pac dots: ", gameState.getNumFood())
        # Manhattan distance to the closest pac dot
        print("Distance nearest pac dots: ", gameState.getDistanceNearestFood())
        # Map walls
        print("Map:")
        print(gameState.getWalls())
        # Score
        print("Score: ", gameState.getScore())

    def chooseAction(self, gameState):

        x = self.getState(gameState)

        move = self.weka.predict("pacdots_model.model", x, "pacdots_training.arff")

        if move not in gameState.getLegalPacmanActions():
            number = random.randint(0, len(gameState.getLegalPacmanActions()) - 1)
            move = gameState.getLegalPacmanActions()[number]

        return move


    def printLineData(self, gameState):
        if not os.path.isfile("dots_kayboard_samemap.arff"):
            self.file = open("dots_kayboard_samemap.arff", "a")
            attributes = ("@relation distance_raw_data\n"
                          "@attribute 'Legal action North' {False, True}\n"
                          "@attribute 'Legal action South' {False, True}\n"
                          "@attribute 'Legal action East' {False, True}\n"
                          "@attribute 'Legal action West' {False, True}\n"
                          "@attribute 'North ghost' numeric\n"
                          "@attribute 'South ghost' numeric\n"
                          "@attribute 'East ghost' numeric\n"
                          "@attribute 'West ghost' numeric\n"
                          "@attribute 'North dot' numeric\n"
                          "@attribute 'South dot' numeric\n"
                          "@attribute 'East dot' numeric\n"
                          "@attribute 'West dot' numeric\n"
                          "@attribute 'Pacman direction' {North, South, East, West, Stop}\n"
                          "@attribute 'Current score' numeric \n"
                          "@attribute 'Future score' numeric \n"
                          "@data\n")

            self.file.write(attributes)

        self.file = open("dots_kayboard_samemap.arff", "a")

        # The information must be a string type, as that is the one that the write method requires
        # For that reason, the concatenation of strings is used with the information we used in the previous method
        # to decide where the pac man should go
        if "North" in gameState.getLegalPacmanActions():
            North = True
        else:
            North = False

        if "South" in gameState.getLegalPacmanActions():
            South = True
        else:
            South = False

        if "East" in gameState.getLegalPacmanActions():
            East = True
        else:
            East = False

        if "West" in gameState.getLegalPacmanActions():
            West = True
        else:
            West = False

        distancer = Distancer(gameState.data.layout)

        min_dist = sys.maxsize

        for i in range(len(gameState.data.ghostDistances)):
            # The ghost is not in jail if the distance is not None
            if gameState.data.ghostDistances[i] != None:
                aux_dist = distancer.getDistance(gameState.getPacmanPosition(), gameState.getGhostPositions()[i])
                if aux_dist < min_dist:
                    min_dist = aux_dist
                    ghost_pos = gameState.getGhostPositions()[i]

        if "North" in gameState.getLegalPacmanActions():
            up = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
            pacmanup = distancer.getDistance(up, ghost_pos)
        else:
            pacmanup = "?"
        if "South" in gameState.getLegalPacmanActions():
            down = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
            pacmandown = distancer.getDistance(down, ghost_pos)
        else:
            pacmandown = "?"
        if "East" in gameState.getLegalPacmanActions():
            right = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
            pacmanright = distancer.getDistance(right, ghost_pos)
        else:
            pacmanright = "?"
        if "West" in gameState.getLegalPacmanActions():
            left = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])
            pacmanleft = distancer.getDistance(left, ghost_pos)

        else:
            pacmanleft = "?"

        # PACDOTS
        pacdots = []
        grid_food = gameState.getFood()
        matrix = []

        for i in range(grid_food.height):
            actual_line = grid_food[i]

            actual_line = actual_line[::-1]

            matrix.append(actual_line)

        num_rows = len(matrix)
        num_cols = len(matrix[0])

        transpose = [[0 for j in range(num_rows)] for i in range(num_cols)]

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                transpose[j][i] = matrix[i][j]

        for i in range(num_rows):
            for j in range(num_cols):
                if transpose[i][j]:
                    place = (i + 1, j + 1)
                    pacdots.append(place)

        min = sys.maxsize
        short_dot = None
        for i in range(len(pacdots)):
            if distancer.getDistance(gameState.getPacmanPosition(), pacdots[i]) < min:
                min = distancer.getDistance(gameState.getPacmanPosition(), pacdots[i])
                short_dot = pacdots[i]

        if short_dot != None:
            if "North" in gameState.getLegalPacmanActions():
                up = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
                north_dot = distancer.getDistance(up, short_dot)
            else:
                north_dot = "?"
            if "South" in gameState.getLegalPacmanActions():
                down = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
                south_dot = distancer.getDistance(down, short_dot)
            else:
                south_dot = "?"
            if "East" in gameState.getLegalPacmanActions():
                right = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
                right_dot = distancer.getDistance(right, short_dot)
            else:
                right_dot = "?"
            if "West" in gameState.getLegalPacmanActions():
                left = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])
                left_dot = distancer.getDistance(left, short_dot)

            else:
                left_dot = "?"
        else:
            north_dot = "?"
            south_dot = "?"
            right_dot = "?"
            left_dot = "?"

        if self.countActions == 1:
            with open("dots_kayboard_samemap.arff", "r") as arff_file:
                lines = arff_file.readlines()
                if lines[-1] != "@data\n":
                    lines.pop()  # remove the last line

            with open("dots_kayboard_samemap.arff", "w") as arff_file:
                arff_file.write(''.join(lines))

            info = str(North) + ',' + str(South) + ',' + str(East) + ',' + str(West) + ',' + \
                   str(pacmanup) + ',' + str(pacmandown) + ',' + str(pacmanright) + ',' + str(pacmanleft) + ',' + \
                   str(north_dot) + ',' + str(south_dot) + ',' + str(right_dot) + ',' + str(left_dot) + ',' + \
                   str(gameState.data.agentStates[0].getDirection()) + ',' + str(gameState.getScore()) + ','

        else:
            info = str(gameState.getScore()) + '\n' + str(North) + ',' + str(South) + ',' + str(East) + ',' + str(
                West) + ',' + \
                   str(north_dot) + ',' + str(south_dot) + ',' + str(right_dot) + ',' + str(left_dot) + ',' + \
                   str(pacmanup) + ',' + str(pacmandown) + ',' + str(pacmanright) + ',' + str(pacmanleft) + ',' + \
                   str(gameState.data.agentStates[0].getDirection()) + ',' + str(gameState.getScore()) + ','

        # Then the previous string is returned

        return info

    def getState(self, gameState):

        if "North" in gameState.getLegalPacmanActions():
            North = True
        else:
            North = False

        if "South" in gameState.getLegalPacmanActions():
            South = True
        else:
            South = False

        if "East" in gameState.getLegalPacmanActions():
            East = True
        else:
            East = False

        if "West" in gameState.getLegalPacmanActions():
            West = True
        else:
            West = False

        distancer = Distancer(gameState.data.layout)

        min_dist = sys.maxsize

        for i in range(len(gameState.data.ghostDistances)):
            # The ghost is not in jail if the distance is not None
            if gameState.data.ghostDistances[i] != None:
                aux_dist = distancer.getDistance(gameState.getPacmanPosition(), gameState.getGhostPositions()[i])
                if aux_dist < min_dist:
                    min_dist = aux_dist
                    ghost_pos = gameState.getGhostPositions()[i]

        if "North" in gameState.getLegalPacmanActions():
            up = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
            pacmanup = distancer.getDistance(up, ghost_pos)
        else:
            pacmanup = "?"
        if "South" in gameState.getLegalPacmanActions():
            down = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
            pacmandown = distancer.getDistance(down, ghost_pos)
        else:
            pacmandown = "?"
        if "East" in gameState.getLegalPacmanActions():
            right = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
            pacmanright = distancer.getDistance(right, ghost_pos)
        else:
            pacmanright = "?"
        if "West" in gameState.getLegalPacmanActions():
            left = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])
            pacmanleft = distancer.getDistance(left, ghost_pos)

        else:
            pacmanleft = "?"

        # PACDOTS
        pacdots = []
        grid_food = gameState.getFood()
        matrix = []

        for i in range(grid_food.height):
            actual_line = grid_food[i]

            actual_line = actual_line[::-1]

            matrix.append(actual_line)

        num_rows = len(matrix)
        num_cols = len(matrix[0])

        transpose = [[0 for j in range(num_rows)] for i in range(num_cols)]

        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                transpose[j][i] = matrix[i][j]

        for i in range(num_rows):
            for j in range(num_cols):
                if transpose[i][j]:
                    place = (i + 1, j + 1)
                    pacdots.append(place)

        min = sys.maxsize
        short_dot = None
        for i in range(len(pacdots)):
            if distancer.getDistance(gameState.getPacmanPosition(), pacdots[i]) < min:
                min = distancer.getDistance(gameState.getPacmanPosition(), pacdots[i])
                short_dot = pacdots[i]

        if short_dot != None:
            if "North" in gameState.getLegalPacmanActions():
                up = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
                north_dot = distancer.getDistance(up, short_dot)
            else:
                north_dot = "?"
            if "South" in gameState.getLegalPacmanActions():
                down = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
                south_dot = distancer.getDistance(down, short_dot)
            else:
                south_dot = "?"
            if "East" in gameState.getLegalPacmanActions():
                right = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
                right_dot = distancer.getDistance(right, short_dot)
            else:
                right_dot = "?"
            if "West" in gameState.getLegalPacmanActions():
                left = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])
                left_dot = distancer.getDistance(left, short_dot)

            else:
                left_dot = "?"
        else:
            north_dot = "?"
            south_dot = "?"
            right_dot = "?"
            left_dot = "?"

        info = [str(North), str(South), str(East), str(West), north_dot, south_dot, right_dot, left_dot, pacmanup,
                pacmandown, pacmanright, pacmanleft]

        # Then the previous string is returned

        return info