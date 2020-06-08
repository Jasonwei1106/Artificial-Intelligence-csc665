# valueIterationAgents.py
# -----------------------
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


# valueIterationAgents.py
# -----------------------
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


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for i in range(self.iterations):
            newValues = util.Counter()
            for state in self.mdp.getStates():
                #if the game is over then the value should be 0
                if self.mdp.isTerminal(state):
                    newValues[state] = 0
                    continue

                maxValue = []
                actionlist = self.mdp.getPossibleActions(state)
                #if there is no action then return 0
                if not actionlist:
                    newValues[state] = 0

                for action in actionlist:
                    #Compute the value of qstate
                    maxValue.append(self.getQValue(state, action))

                    # Find the maximum action
                newValues[state] = max(maxValue)

            self.values = newValues


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        #each time in the iteration, we sum the qvalue
        actionPrime = 0
        for transition in self.mdp.getTransitionStatesAndProbs(state, action):
            statePrime = transition[0]
            prob = transition[1]
            gamma = self.discount
            #R(s,a,stateprime)
            reward = self.mdp.getReward(state, action, statePrime)
            #calculate the qstar(s)
            actionPrime += prob * (reward + (gamma * self.values[statePrime]))

        return actionPrime

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        #it will be the same as computeqvalue but this one just return the action
        maxValue = float('-inf')
        maxAction = None
        actionlist = self.mdp.getPossibleActions(state)
        # if there are no legal actions you should return None.
        if not actionlist or self.mdp.isTerminal(state):
            return None

        for action in actionlist:
            actionqvalue = self.getQValue(state, action)

            # Find the maximum action
            if maxValue < actionqvalue:
                maxAction = action
                maxValue = actionqvalue

        return maxAction


    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        #"we will update only one state in each iteration"
        states = self.mdp.getStates()
        count = 0
        for i in range(self.iterations):
            #update only one state in each iteration
            state = states[count]
            if count >= len(states)-1:
                count = 0
            else:
                count += 1
            #every time do the iteration, make a copy of values
            maxValue = []
            if self.mdp.isTerminal(state):
                self.values[state] = 0

            else:
                for action in self.mdp.getPossibleActions(state):
                    actionValue = 0.0

                    #Do the calculation Sum of T(s,a,sprime)*[reward[s,a,sprime] + discount * value(sprime)]
                    for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                        gamma = self.discount
                        statePrime = transition[0]
                        prob = transition[1]
                        actionValue += prob * (self.mdp.getReward(state, action, statePrime) + gamma * self.values[statePrime])
                    maxValue.append(actionValue)
                self.values[state] = max(maxValue)




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        # Initialize an empty set so no duplicate
        predecessors = {}
        # initialize an empty priority queue
        pqueue = util.PriorityQueue()

        # For each state s do
        for state in self.mdp.getStates():

            # Compute predecessors of all states
            #we define the predecessors of a state s as all states that have a nonzero
            # probability of reaching s by taking some action a
            for action in self.mdp.getPossibleActions(state):
                #predecessor will be the state based prime state
                for transition in self.mdp.getTransitionStatesAndProbs(state, action):
                    if not transition[0] in predecessors:
                        predecessors[transition[0]] = [state]
                    elif transition[0] in predecessors:
                        predecessors[transition[0]].append(state)

            # if terminal state then 0
                if self.mdp.isTerminal(state):
                    self.values[state] = 0

                # For each non-terminal state s, do
                else:
                    # Get highest possible Q value from state
                    maxValues = []
                    for action in self.mdp.getPossibleActions(state):
                        qvalue = self.computeQValueFromValues(state, action)
                        maxValues.append(qvalue)
                diff = abs(self.values[state] - max(maxValues))
                # push s into the priority queue qith priority -diff
                pqueue.update(state, -diff)

        for i in range(self.iterations):
            # if the priority queue is empty, then terminate
            if pqueue.isEmpty():
                break
            else:
                # Pop a state off the priority queue
                state = pqueue.pop()

                # If not terminal, update state's value in self.values
                if not self.mdp.isTerminal(state):
                    maxValues = []
                    for action in self.mdp.getPossibleActions(state):
                        qvalue = self.computeQValueFromValues(state,action)
                        maxValues.append(qvalue)
                    self.values[state] = max(maxValues)
                # for each predecessor pred of state:
                for predecessor in predecessors[state]:
                    if not self.mdp.isTerminal(predecessor):
                    # Get highest possible Q value from state
                        maxValues = []
                        for action in self.mdp.getPossibleActions(predecessor):
                            qvalue = self.computeQValueFromValues(predecessor,action)
                            maxValues.append(qvalue)
                        diff = abs(self.values[predecessor] - max(maxValues))
                        if diff > self.theta:
                            pqueue.update(predecessor, -diff)



