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
        for i in range(0, self.iterations):
            allValues = util.Counter()
            for state in self.mdp.getStates():
                values = []
                if self.mdp.isTerminal(state):
                    values = [0]

                for action in self.mdp.getPossibleActions(state):
                    values.append(self.computeQValueFromValues(state, action))

                allValues[state] = max(values)

            self.values = allValues


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
        Qval = 0

        for i in self.mdp.getTransitionStatesAndProbs(state, action):
            Qval += i[1] * (self.mdp.getReward(state,action,i[0]) + (self.discount * self.values[i[0]]) )

        return Qval

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        successors = self.mdp.getPossibleActions(state)

        if not successors or self.mdp.isTerminal(state):
            return None

        action = None
        maxVal = None

        for i in successors:
            QVal = self.computeQValueFromValues(state, i)

            #print(maxVal is None or maxVal < QVal)
            if maxVal is None or maxVal < QVal:
                maxVal = QVal
                action = i


        return action

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
        ct = 0

        while ct < self.iterations:
            for state in self.mdp.getStates():
                allValues = util.Counter()

                for action in self.mdp.getPossibleActions(state):
                    allValues[action] = (self.computeQValueFromValues(state, action))

                self.values[state] = allValues[allValues.argMax()]

                ct+=1
                if ct >= self.iterations:
                    return

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
        "*** YOUR CODE HERE ***"
        def normalVal(state):
            return max(self.getQValue(state, a) for a in self.mdp.getPossibleActions(state))

        priorityQ = util.PriorityQueue()

        parents = {}
        for state in self.mdp.getStates():
            parents[state] = set()
            self.values[state] = 0

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                allValues = util.Counter()
                for successor in self.mdp.getPossibleActions(state):
                    for i in self.mdp.getTransitionStatesAndProbs(state, successor):
                        if i[1] > 0:
                            parents[i[0]].add(state)

            if not self.mdp.isTerminal(state):
                diff = abs(self.values[state] - normalVal(state))
                priorityQ.push(state, -diff)


        for i in range(0, self.iterations):
            if priorityQ.isEmpty():
                break

            currState = priorityQ.pop()

            self.values[currState] = normalVal(currState)

            for p in parents[currState]:

                diff = abs(self.values[p] - normalVal(p))
                if diff > self.theta:
                    priorityQ.update(p, -diff)
