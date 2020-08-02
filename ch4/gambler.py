from typing import List
import matplotlib.pyplot as plt

'''
Approach:

Note that states are defined by a single positive integer, the amount of money
Actions are also defined by a single positive integer, the kh
'''


class GamblerState:
    '''
    GamblerState gives the actions associated with a state
    '''
    def __init__(self, i: int, N: int):
        self.i = i
        self.N = N

    def getActions(self) -> List[int]:
        i, N = self.i, self.N
        return list(range(1, min(N - i, i) + 1))

    def __str__(self):
        return 'state i {} N {} actions {}'.format(self.i, self.N, self.getActions())

class GamblerSingleton:
    def __init__(self, initial: float, theta: float, gamma: float, N: int, pHeads: float):
        self.theta = theta
        self.gamma = gamma
        self.pHeads = pHeads
        self.states = [GamblerState(i, N) for i in range(1, N)]
        self.N = N
        self.values = [initial] * (N + 1)
        self.values[0] = 0
        self.values[N] = 1

    def getValue(self, state: GamblerState) -> float:
        return self.values[state.i]

    def setValue(self, state: GamblerState) -> float:
        actions = state.getActions()
        maxValue = self.getActionStateSum(state, actions[0])
        for action in actions[1:]:
            maxValue = max(maxValue, self.getActionStateSum(state, action))
        self.values[state.i] = maxValue
        return maxValue

    def getPolicy(self) -> List[int]:
        policy = []
        for state in self.states:
            actions = state.getActions()
            maxAction, maxVal = actions[0], self.getActionStateSum(state, actions[0])
            for action in actions[1:]:
                curVal = self.getActionStateSum(state, action)
                if curVal > maxVal:
                    maxVal = curVal
                    maxAction = action
            policy.append(maxAction)
        return policy

    def getActionStateSum(self, state: GamblerState, action: int) -> float:
        pHeads, gamma = self.pHeads, self.gamma
        headsIndex = state.i + action
        tailsIndex = state.i - action

        reward = 0
        if headsIndex == self.N:
            reward = 1
        return pHeads * (reward + gamma * self.values[headsIndex]) + (1 - pHeads) * (gamma * self.values[tailsIndex])

    def valueIteration(self) -> List[int]:
        theta, gamma = self.theta, self.gamma
        delta, i = 0.5, 0
        while True:
            print('iteration {} delta {} theta {}'.format(i, delta, theta))
            for state in self.states:
                curVal = self.getValue(state)
                newValue = self.setValue(state)
                delta = min(delta, abs(curVal - newValue))
            if delta < theta:
                break
            i += 1
        return self.getPolicy()

    def graph(self):
        names = range(1, self.N)
        values = self.getPolicy()

        fig, axs = plt.subplots(1, 1, figsize=(9, 3), sharey=True)
        axs.bar(names, values)
        # axs[1].scatter(names, values)
        # axs[2].plot(names, values)
        fig.suptitle('Gambler pHeads {} theta {} gamma {}'.format(self.pHeads, self.theta, self.gamma))
        plt.show()

if __name__ == '__main__':
    gs = GamblerSingleton(0.5, 0.00001, 0.4, 100, 0.4)
    testStates = [GamblerState(i, 100) for i in [1, 40, 50, 51, 70, 99]]
    for testState in testStates:
        print('testState {}'.format(testState))
        assert gs.getValue(testState) == 0.5

    print(gs.valueIteration())
    gs.graph()
