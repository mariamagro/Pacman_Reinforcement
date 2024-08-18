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
        if not os.path.isfile("distance_raw_data.arff"):
            self.file = open("distance_raw_data.arff", "a")
            attributes = ("@relation all_data_pacman\n"
                          "@attribute 'Legal action North' {False, True}\n"
                          "@attribute 'Legal action South' {False, True}\n"
                          "@attribute 'Legal action East' {False, True}\n"
                          "@attribute 'Legal action West' {False, True}\n"
                          "@attribute 'North ghost 1' numeric\n"
                          "@attribute 'South ghost 1' numeric\n"
                          "@attribute 'East ghost 1' numeric\n"
                          "@attribute 'West ghost 1' numeric\n"
                          "@attribute 'North ghost 2' numeric\n"
                          "@attribute 'South ghost 2' numeric\n"
                          "@attribute 'East ghost 2' numeric\n"
                          "@attribute 'West ghost 2' numeric\n"
                          "@attribute 'North ghost 3' numeric\n"
                          "@attribute 'South ghost 3' numeric\n"
                          "@attribute 'East ghost 3' numeric\n"
                          "@attribute 'West ghost 3' numeric\n"
                          "@attribute 'North ghost 4' numeric\n"
                          "@attribute 'South ghost 4' numeric\n"
                          "@attribute 'East ghost 4' numeric\n"
                          "@attribute 'West ghost 4' numeric\n"
                          "@attribute 'Pacman direction' {North, South, East, West, Stop}\n"
                          "@attribute 'Current score' numeric \n"
                          "@attribute 'Future score' numeric \n"
                          "@data\n")

            self.file.write(attributes)

        self.file = open("distance_raw_data.arff", "a")

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

        # Future possible positions of pacman

        pacmanup = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
        pacmandown = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
        pacmanright = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
        pacmanleft = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])

        # Distances at north position

        if "North" in gameState.getLegalPacmanActions():
            north_ghost1 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[0])
            north_ghost2 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[1])
            north_ghost3 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[2])
            north_ghost4 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[3])

        else:
            north_ghost1 = "?"
            north_ghost2 = "?"
            north_ghost3 = "?"
            north_ghost4 = "?"

        if "South" in gameState.getLegalPacmanActions():
            south_ghost1 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[0])
            south_ghost2 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[1])
            south_ghost3 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[2])
            south_ghost4 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[3])
        else:
            south_ghost1 = "?"
            south_ghost2 = "?"
            south_ghost3 = "?"
            south_ghost4 = "?"

        if "East" in gameState.getLegalPacmanActions():
            east_ghost1 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[0])
            east_ghost2 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[1])
            east_ghost3 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[2])
            east_ghost4 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[3])
        else:
            east_ghost1 = "?"
            east_ghost2 = "?"
            east_ghost3 = "?"
            east_ghost4 = "?"

        if "West" in gameState.getLegalPacmanActions():
            west_ghost1 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[0])
            west_ghost2 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[1])
            west_ghost3 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[2])
            west_ghost4 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[3])
        else:
            west_ghost1 = "?"
            west_ghost2 = "?"
            west_ghost3 = "?"
            west_ghost4 = "?"

        width, height = gameState.data.layout.width, gameState.data.layout.height

        if north_ghost1 is not "?" and (north_ghost1 is None or north_ghost1 > width * height):
            north_ghost1 = "?"

        if south_ghost1 is not "?" and (south_ghost1 is None or south_ghost1 > width * height):
            south_ghost1 = "?"

        if east_ghost1 is not "?" and (east_ghost1 is None or east_ghost1 > width * height):
            east_ghost1 = "?"

        if west_ghost1 is not "?" and (west_ghost1 is None or west_ghost1 > width * height):
            west_ghost1 = "?"

        if north_ghost2 is not "?" and (north_ghost2 is None or north_ghost2 > width * height):
            north_ghost2 = "?"

        if south_ghost2 is not "?" and (south_ghost2 is None or south_ghost2 > width * height):
            south_ghost2 = "?"

        if east_ghost2 is not "?" and (east_ghost2 is None or east_ghost2 > width * height):
            east_ghost2 = "?"

        if west_ghost2 is not "?" and (west_ghost2 is None or west_ghost2 > width * height):
            west_ghost2 = "?"

        if north_ghost3 is not "?" and (north_ghost3 is None or north_ghost3 > width * height):
            north_ghost3 = "?"

        if south_ghost3 is not "?" and (south_ghost3 is None or south_ghost3 > width * height):
            south_ghost3 = "?"

        if east_ghost3 is not "?" and (east_ghost3 is None or east_ghost3 > width * height):
            east_ghost3 = "?"

        if west_ghost3 is not "?" and (west_ghost3 is None or west_ghost3 > width * height):
            west_ghost3 = "?"

        if north_ghost4 is not "?" and (north_ghost4 is None or north_ghost4 > width * height):
            north_ghost4 = "?"

        if south_ghost4 is not "?" and (south_ghost4 is None or south_ghost4 > width * height):
            south_ghost4 = "?"

        if east_ghost4 is not "?" and (east_ghost4 is None or east_ghost4 > width * height):
            east_ghost4 = "?"

        if west_ghost4 is not "?" and (west_ghost4 is None or west_ghost4 > width * height):
            west_ghost4 = "?"

        if self.countActions == 1:
            with open("distance_raw_data.arff", "r") as arff_file:
                lines = arff_file.readlines()
                if lines[-1] != "@data\n":
                    lines.pop()  # remove the last line

            with open("distance_raw_data.arff", "w") as arff_file:
                arff_file.write(''.join(lines))

            info = str(North) + ',' + str(South) + ',' + str(East) + ',' + str(West) + ',' + \
                   str(north_ghost1) + ',' + str(south_ghost1) + ',' + str(east_ghost1) + ',' + str(west_ghost1) + ',' + \
                   str(north_ghost2) + ',' + str(south_ghost2) + ',' + str(east_ghost2) + ',' + str(west_ghost2) + ',' + \
                   str(north_ghost3) + ',' + str(south_ghost3) + ',' + str(east_ghost3) + ',' + str(west_ghost3) + ',' + \
                   str(north_ghost4) + ',' + str(south_ghost4) + ',' + str(east_ghost4) + ',' + str(west_ghost4) + ',' + \
                   str(gameState.data.agentStates[0].getDirection()) + ',' + str(gameState.getScore()) + ','

        else:

            info = str(gameState.getScore()) + '\n' + str(North) + ',' + str(South) + ',' + str(East) + ',' + str(West) + ',' + \
                   str(north_ghost1) + ',' + str(south_ghost1) + ',' + str(east_ghost1) + ',' + str(west_ghost1) + ',' + \
                   str(north_ghost2) + ',' + str(south_ghost2) + ',' + str(east_ghost2) + ',' + str(west_ghost2) + ',' + \
                   str(north_ghost3) + ',' + str(south_ghost3) + ',' + str(east_ghost3) + ',' + str(west_ghost3) + ',' + \
                   str(north_ghost4) + ',' + str(south_ghost4) + ',' + str(east_ghost4) + ',' + str(west_ghost4) + ',' + \
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

        move = self.weka.predict("model_ghost_distances.model", x, "training_distance.arff")

        # Random movement if the action is illegal

        if move not in gameState.getLegalPacmanActions():
            number = random.randint(0, len(gameState.getLegalPacmanActions()) - 1)
            move = gameState.getLegalPacmanActions()[number]


        return move


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

        # Future possible positions of pacman

        pacmanup = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] + 1)
        pacmandown = (gameState.getPacmanPosition()[0], gameState.getPacmanPosition()[1] - 1)
        pacmanright = (gameState.getPacmanPosition()[0] + 1, gameState.getPacmanPosition()[1])
        pacmanleft = (gameState.getPacmanPosition()[0] - 1, gameState.getPacmanPosition()[1])

        # Distances at north position

        if "North" in gameState.getLegalPacmanActions():
            north_ghost1 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[0])
            north_ghost2 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[1])
            north_ghost3 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[2])
            north_ghost4 = distancer.getDistance(pacmanup, gameState.getGhostPositions()[3])

        else:
            north_ghost1 = "?"
            north_ghost2 = "?"
            north_ghost3 = "?"
            north_ghost4 = "?"

        if "South" in gameState.getLegalPacmanActions():
            south_ghost1 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[0])
            south_ghost2 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[1])
            south_ghost3 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[2])
            south_ghost4 = distancer.getDistance(pacmandown, gameState.getGhostPositions()[3])
        else:
            south_ghost1 = "?"
            south_ghost2 = "?"
            south_ghost3 = "?"
            south_ghost4 = "?"

        if "East" in gameState.getLegalPacmanActions():
            east_ghost1 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[0])
            east_ghost2 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[1])
            east_ghost3 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[2])
            east_ghost4 = distancer.getDistance(pacmanright, gameState.getGhostPositions()[3])
        else:
            east_ghost1 = "?"
            east_ghost2 = "?"
            east_ghost3 = "?"
            east_ghost4 = "?"

        if "West" in gameState.getLegalPacmanActions():
            west_ghost1 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[0])
            west_ghost2 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[1])
            west_ghost3 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[2])
            west_ghost4 = distancer.getDistance(pacmanleft, gameState.getGhostPositions()[3])
        else:
            west_ghost1 = "?"
            west_ghost2 = "?"
            west_ghost3 = "?"
            west_ghost4 = "?"

        width, height = gameState.data.layout.width, gameState.data.layout.height

        if north_ghost1 is not "?" and (north_ghost1 is None or north_ghost1 > width * height):
            north_ghost1 = "?"

        if south_ghost1 is not "?" and (south_ghost1 is None or south_ghost1 > width * height):
            south_ghost1 = "?"

        if east_ghost1 is not "?" and (east_ghost1 is None or east_ghost1 > width * height):
            east_ghost1 = "?"

        if west_ghost1 is not "?" and (west_ghost1 is None or west_ghost1 > width * height):
            west_ghost1 = "?"

        if north_ghost2 is not "?" and (north_ghost2 is None or north_ghost2 > width * height):
            north_ghost2 = "?"

        if south_ghost2 is not "?" and (south_ghost2 is None or south_ghost2 > width * height):
            south_ghost2 = "?"

        if east_ghost2 is not "?" and (east_ghost2 is None or east_ghost2 > width * height):
            east_ghost2 = "?"

        if west_ghost2 is not "?" and (west_ghost2 is None or west_ghost2 > width * height):
            west_ghost2 = "?"

        if north_ghost3 is not "?" and (north_ghost3 is None or north_ghost3 > width * height):
            north_ghost3 = "?"

        if south_ghost3 is not "?" and (south_ghost3 is None or south_ghost3 > width * height):
            south_ghost3 = "?"

        if east_ghost3 is not "?" and (east_ghost3 is None or east_ghost3 > width * height):
            east_ghost3 = "?"

        if west_ghost3 is not "?" and (west_ghost3 is None or west_ghost3 > width * height):
            west_ghost3 = "?"

        if north_ghost4 is not "?" and (north_ghost4 is None or north_ghost4 > width * height):
            north_ghost4 = "?"

        if south_ghost4 is not "?" and (south_ghost4 is None or south_ghost4 > width * height):
            south_ghost4 = "?"

        if east_ghost4 is not "?" and (east_ghost4 is None or east_ghost4 > width * height):
            east_ghost4 = "?"

        if west_ghost4 is not "?" and (west_ghost4 is None or west_ghost4 > width * height):
            west_ghost4 = "?"

        info = [str(North), str(South), str(East), str(West), north_ghost1, south_ghost1, east_ghost1,
                west_ghost1, north_ghost2, south_ghost2, east_ghost2, west_ghost2, north_ghost3, south_ghost3,
                east_ghost3, west_ghost3, north_ghost4, south_ghost4, east_ghost4, west_ghost4]

        return info

