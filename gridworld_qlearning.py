import numpy as np
import random

# Grid size
rows, cols = 4, 4

# Rewards
goal = (3, 3)
penalties = [(0,3), (1,1), (3,0)]
psuh
# Q-table
Q = np.zeros((rows, cols, 4))

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 500

actions = [( -1,0), (1,0), (0,-1), (0,1)]  # up, down, left, right

def get_reward(state):
    if state == goal:
        return 10
    elif state in penalties:
        return -10
    else:
        return -1

def is_valid(r, c):
    return 0 <= r < rows and 0 <= c < cols

for ep in range(episodes):
    state = (0,0)

    while state != goal:
        r, c = state

        # epsilon-greedy
        if random.uniform(0,1) < epsilon:
            action = random.randint(0,3)
        else:
            action = np.argmax(Q[r,c])

        dr, dc = actions[action]
        nr, nc = r + dr, c + dc

        if not is_valid(nr, nc):
            nr, nc = r, c

        reward = get_reward((nr,nc))

        # Q update
        Q[r,c,action] = Q[r,c,action] + alpha * (
            reward + gamma * np.max(Q[nr,nc]) - Q[r,c,action]
        )

        state = (nr,nc)

print("Learned Policy:")
for i in range(rows):
    for j in range(cols):
        print(np.argmax(Q[i,j]), end=" ")
    print()
