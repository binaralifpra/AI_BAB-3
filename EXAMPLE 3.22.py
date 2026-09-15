#Example 3.22 Q-Learning
import numpy as np
import pylab as plt
import networkx as nx

def showgraph(points_list):
    G=nx.Graph()
    G.add_edges_from(points_list)
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G,pos)
    nx.draw_networkx_edges(G,pos)
    nx.draw_networkx_labels(G,pos)
    plt.show()

def createRmat(MATRIX_SIZE,points_list,goal):
    R = np.matrix(np.ones(shape=(MATRIX_SIZE, MATRIX_SIZE)))
    R *= -1
    for point in points_list:
        if point[1] == goal:
            R[point] = 100
        else:
            R[point] = 0
        if point[0] == goal:
            R[point[::-1]] = 100
        else:
            R[point[::-1]]= 0
    R[goal,goal]= 100
    return R

def available_actions(R, state):
    current_state_row = R[state,]
    av_act = np.where(current_state_row >= 0)[1]
    return av_act

def sample_next_action(available_act):
    next_action = int(np.random.choice(available_act,1))
    return next_action

def update(R, Q, current_state, action, gamma):
    max_index = np.where(Q[action,] == np.max(Q[action,]))[1]
    if max_index.shape[0] > 1:
        max_index = int(np.random.choice(max_index, size = 1))
    else:
        max_index = int(max_index)
    max_value = Q[action, max_index]
    Q[current_state, action] = R[current_state, action] + gamma * max_value
    if (np.max(Q) > 0):
        return(np.sum(Q/np.max(Q)*100))
    else:
        return (0)

points_list = [(0,1),(1,2),(1,3),(2,4),(3,5),(3,6)]
goal = 6
# showgraph(points_list) # Un-comment to see plot
MATRIX_SIZE = 7
R = createRmat(MATRIX_SIZE,points_list,goal)
Q = np.matrix(np.zeros([MATRIX_SIZE,MATRIX_SIZE]))
gamma = 0.8
scores = []
for i in range(700):
    current_state = np.random.randint(0, int(Q.shape[0]))
    available_act = available_actions(R, current_state)
    action = sample_next_action(available_act)
    score = update(R,Q,current_state,action,gamma)
    scores.append(score)

print("Trained Q matrix:")
print(Q/np.max(Q)*100)

current_state = 0
steps = [current_state]
while current_state != goal:
    next_step_index = np.where(Q[current_state,] == np.max(Q[current_state,]))[1]
    if next_step_index.shape[0] > 1:
        next_step_index = int(np.random.choice(next_step_index, size=1))
    else:
        next_step_index = int(next_step_index)
    steps.append(next_step_index)
    current_state = next_step_index

print("Most efficient path:")
print(steps)
