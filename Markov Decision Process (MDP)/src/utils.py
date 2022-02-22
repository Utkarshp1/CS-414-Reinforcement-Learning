import numpy as np

def generate_MDP(x, y, goal_coordinates, rewards, prob, wall_coor):
    n = x*y
    m = 4
    trans_prob = np.zeros((n, n, m))
    calculate_up_prob(trans_prob, x, y, prob, wall_coor)
    calculate_down_prob(trans_prob, x, y, prob, wall_coor)
    calculate_left_prob(trans_prob, x, y, prob, wall_coor)
    calculate_right_prob(trans_prob, x, y, prob, wall_coor)
    
    wall_index = wall_coor[1]*x + wall_coor[0]  
    for i in range(m):
        for j in range(n):
            trans_prob[wall_index, j, i] = 0
        for j in range(n):
            trans_prob[j, wall_index, i] = 0
            
    reward = np.full((n, n, m), rewards['general'])
    pos_goal_state_index = (goal_coordinates['positive'][1]*x + 
                            goal_coordinates['positive'][0])
    neg_goal_state_index = (goal_coordinates['negative'][1]*x + 
                            goal_coordinates['negative'][0])
    reward[pos_goal_state_index, :, :] = rewards['positive']
    reward[neg_goal_state_index, :, :] = rewards['negative']

    # for i in range(m):
    #     is_goal_trans = trans_prob[:, pos_goal_state_index, i] != 0
    #     # print(is_goal_trans.shape)
    #     # print(is_goal_trans)
    #     reward[is_goal_trans, pos_goal_state_index, i] = rewards['positive']

    #     is_goal_trans = trans_prob[:, neg_goal_state_index, i] != 0
    #     reward[is_goal_trans, neg_goal_state_index, i] = rewards['negative']
    
    return trans_prob, reward
    
    
def calculate_up_prob(trans_prob, x, y, prob, wall_coor):
    for i in range(trans_prob.shape[0]):
        state_coor = (i%x, i//x)
        
        if state_coor[1] == 0:
            trans_prob[i, i, 0] = prob['success_prob']
        elif (state_coor[0] == wall_coor[0] and state_coor[1] == wall_coor[1] + 1):
            trans_prob[i, i, 0] = prob['success_prob']
        else:
            trans_prob[i, i-x, 0] = prob["success_prob"]
        
        if state_coor[0] == 0:
            trans_prob[i, i, 0] += prob['left_prob']
        elif (state_coor[0] == wall_coor[0] + 1 and state_coor[1] == wall_coor[1]):
            trans_prob[i, i, 0] += prob['left_prob']
        else:
            trans_prob[i, i-1, 0] = prob['left_prob']
            
        if state_coor[0] == x-1:
            trans_prob[i, i, 0] += prob['right_prob']
        elif (state_coor[0] == wall_coor[0] - 1 and state_coor[1] == wall_coor[1]):
            trans_prob[i, i, 0] += prob['right_prob']
        else:
            trans_prob[i, i+1, 0] = prob['right_prob']
            
def calculate_down_prob(trans_prob, x, y, prob, wall_coor):
    for i in range(trans_prob.shape[0]):
        state_coor = (i%x, i//x)
        
        if state_coor[1] == y-1:
            trans_prob[i, i, 1] = prob['success_prob']
        elif (state_coor[0] == wall_coor[0] and state_coor[1] == wall_coor[1] - 1):
            trans_prob[i, i, 1] = prob['success_prob']
        else:
            trans_prob[i, i+x, 1] = prob["success_prob"]
            
        if state_coor[0] == 0:
            trans_prob[i, i, 1] += prob['left_prob']
        elif (state_coor[0] == wall_coor[0] + 1 and state_coor[1] == wall_coor[1]):
            trans_prob[i, i, 1] += prob['left_prob']
        else:
            trans_prob[i, i-1, 1] = prob['left_prob']
            
        if state_coor[0] == x-1:
            trans_prob[i, i, 1] += prob['right_prob']
        elif (state_coor[0] == wall_coor[0] - 1 and state_coor[1] == wall_coor[1]):
            trans_prob[i, i, 1] += prob['right_prob']
        else:
            trans_prob[i, i+1, 1] = prob['right_prob']
        
def calculate_left_prob(trans_prob, x, y, prob, wall_coor):
    for i in range(trans_prob.shape[0]):
        state_coor = (i%x, i//x)
        
        if state_coor[0] == 0:
            trans_prob[i, i, 2] = prob['success_prob']
        elif (state_coor[0] == wall_coor[0] + 1 and state_coor[1] == wall_coor[1]):
            trans_prob[i, i, 2] = prob['success_prob']
        else:
            trans_prob[i, i-1, 2] = prob["success_prob"]
            
        if state_coor[1] == y-1:
            trans_prob[i, i, 2] += prob['left_prob']
        elif state_coor[0] == wall_coor[0] and state_coor[1] == wall_coor[1] - 1:
            trans_prob[i, i, 2] += prob['left_prob']
        else:
            trans_prob[i, i+x, 2] = prob['left_prob']
            
        if state_coor[1] == 0:
            trans_prob[i, i, 2] += prob['right_prob']
        elif state_coor[0] == wall_coor[0] and state_coor[1] == wall_coor[1] + 1:
            trans_prob[i, i, 2] += prob['right_prob']
        else:
            trans_prob[i, i-x, 2] = prob['right_prob']
            
def calculate_right_prob(trans_prob, x, y, prob, wall_coor):
    for i in range(trans_prob.shape[0]):
        state_coor = (i%x, i//x)
        
        if state_coor[0] == x-1:
            trans_prob[i, i, 3] = prob['success_prob']
        elif state_coor[0] == wall_coor[0] - 1 and state_coor[1] == wall_coor[1]:
            trans_prob[i, i, 3] = prob['success_prob']
        else:
            trans_prob[i, i+1, 3] = prob["success_prob"]
            
        if state_coor[1] == 0:
            trans_prob[i, i, 3] += prob['left_prob']
        elif state_coor[0] == wall_coor[0] and state_coor[1] == wall_coor[1] + 1:
            trans_prob[i, i, 3] += prob['left_prob']
        else:
            trans_prob[i, i-x, 3] = prob['left_prob']
            
        if state_coor[1] == y-1:
            trans_prob[i, i, 3] += prob['right_prob']
        elif state_coor[0] == wall_coor[0] and state_coor[1] == wall_coor[1] - 1:
            trans_prob[i, i, 3] += prob['right_prob']
        else:
            trans_prob[i, i+x, 3] = prob['right_prob']
        