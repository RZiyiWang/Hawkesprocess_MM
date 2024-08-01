from tf_agents.environments import tf_py_environment
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from tqdm import tqdm
from tf_agents.policies import random_tf_policy

def converter(x):
    return tf.convert_to_tensor([x])

def evaluation(policy, name,env,calculate_ratio, num_episodes=1000,num_times=10):

    terminal_wealth = []
    terminal_inv = []
    notquote_times =0
    askandbid_times =0
    onlyask_times =0
    onlybid_times =0
    act = [] #for 2actions:[1,0,1...];for 4actions:[1,0,2,3,4,1...]

    environment = tf_py_environment.TFPyEnvironment(env)

    for _ in tqdm(range(num_times)):

        for _ in range(num_episodes):

            time_step = environment.reset()
            policy_state = policy.get_initial_state(batch_size=1)
            
            while not time_step.is_last():
                policy_step = policy.action(time_step, policy_state)
                act.append(policy_step.action.numpy()[0])
                policy_state = policy_step.state
                time_step = environment.step(policy_step.action)
            terminal_inv.append(time_step.observation.numpy()[0][1])
            terminal_wealth.append(time_step.observation.numpy()[0][2])
            
    if calculate_ratio is True:
        for i in act: # 0 -> not quote; 1 -> quote(ask&bid); 2 -> only ask; 3 -> only bid
            if i == 1: #quote(ask&bid)
                askandbid_times +=1
            elif i == 0: #not quote
                notquote_times += 1
            elif i ==2: #quote(ask)
                onlyask_times +=1
            elif i ==3: #quote(bid)
                onlybid_times +=1
            

    average_wealth=np.average(terminal_wealth)
    variance_wealth=np.std(terminal_wealth)
    if variance_wealth != 0:
        sharpe_ratio=average_wealth/variance_wealth
    else:
        sharpe_ratio='variance_wealth=0'
    min_wealth=min(terminal_wealth)
    max_wealth=max(terminal_wealth)

    average_inventory=np.average(terminal_inv)
    variance_inventory=np.std(terminal_inv)

    if calculate_ratio is True:
        total_times=notquote_times+askandbid_times+onlyask_times+onlybid_times
        if total_times !=0:
            notquote_ratio=notquote_times/total_times
            askandbid_ratio=askandbid_times/total_times
            onlyask_ratio=onlyask_times/total_times
            onlybid_ratio=onlybid_times/total_times

            print('The following are the results of '+name+' evaluation:',
                '\naverage wealth:',average_wealth,
                '\nvariance wealth:',variance_wealth,
                '\nsharpe ratio:',sharpe_ratio,
                '\nmin wealth:',min_wealth,
                '\nmax wealth:',max_wealth,
                '\naverage inventory:',average_inventory,
                '\nvariance inventory:',variance_inventory,
                '\nnotquote_ratio: {:.2%}'.format(notquote_ratio),
                '\naskandbid_ratio: {:.2%}'.format(askandbid_ratio),
                '\nonlyask_ratio: {:.2%}'.format(onlyask_ratio),
                '\nonlybid_ratio: {:.2%}'.format(onlybid_ratio),flush=True)
        else:
            print('The following are the random policy results of ' +name+ ' :',
                '\naverage wealth:',average_wealth,
                '\nvariance wealth:',variance_wealth,
                '\nsharpe ratio:',sharpe_ratio,
                '\nmin wealth:',min_wealth,
                '\nmax wealth:',max_wealth,
                '\naverage inventory:',average_inventory,
                '\nvariance inventory:',variance_inventory,
                '\ntotal_times is 0',flush=True)
    else:
        print('The following are the results of '+name+' evaluation:',
            '\naverage wealth:',average_wealth,
            '\nvariance wealth:',variance_wealth,
            '\nsharpe ratio:',sharpe_ratio,
            '\nmin wealth:',min_wealth,
            '\nmax wealth:',max_wealth,
            '\naverage inventory:',average_inventory,
            '\nvariance inventory:',variance_inventory,flush=True)
  
    return terminal_wealth


def validate_with_random_policy(name,env,num_episodes=1000,num_times=10): 
    random_environment = tf_py_environment.TFPyEnvironment(env)
    random_policy = random_tf_policy.RandomTFPolicy(random_environment.time_step_spec(), random_environment.action_spec())
  
    random_wealth_list = []
    terminal_inv = []

    for _ in tqdm(range(num_times)):

        for _ in range(num_episodes):

            time_step = random_environment.reset()
            
            while not time_step.is_last():
                action_step = random_policy.action(time_step)
                time_step = random_environment.step(action_step.action)

            terminal_inv.append(time_step.observation.numpy()[0][1])
            random_wealth_list.append(time_step.observation.numpy()[0][2])

    average_wealth=np.average(random_wealth_list)
    variance_wealth=np.std(random_wealth_list)
    sharpe_ratio=average_wealth/variance_wealth
    min_wealth=min(random_wealth_list)
    max_wealth=max(random_wealth_list)

    average_inventory=np.average(terminal_inv)
    variance_inventory=np.std(terminal_inv)

    print('The following are the random policy results of ' +name+ ' :',
        '\naverage wealth:',average_wealth,
        '\nvariance wealth:',variance_wealth,
        '\nsharpe ratio:',sharpe_ratio,
        '\nmin wealth:',min_wealth,
        '\nmax wealth:',max_wealth,
        '\naverage inventory:',average_inventory,
        '\nvariance inventory:',variance_inventory,flush=True)

  
    return random_wealth_list