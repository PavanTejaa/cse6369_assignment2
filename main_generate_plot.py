import argparse
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_name', '-xn', type=str, default='my_exp', help='name of the experiment')
    # This is for CartPole experiment. It has to tasks to do. 0 - To compare T0, T1, T2 and 1 - To compare T3, T4  
    parser.add_argument('--exp_type', '-xt', type=str, default='0', help='name of the experiment')
    args = parser.parse_args()
    params = vars(args)

    # Load pkl-file containing the learning (reward) history
    exp_name = params['exp_name']
    exp_type = params['exp_type']
    # get the current directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    extension = '.pkl'
    if(exp_type == '0'):
        file_list = [f for f in os.listdir(dir_path) if f.endswith(extension) and exp_name in f and ('t0' in f or 't1' in f or 't2' in f)]
        file_name = ''
        file_list.sort()
        # Plot the data
        for file in file_list:
            with open(file, 'rb') as f:
                values_list = pickle.load(f)
                label = 'T0' if 't0' in file else 'T1' if 't1' in file else 'T2' if 't2' in file else file
                sns.lineplot(data=values_list, linestyle='--', label=label)
    else:
        file_list = [f for f in os.listdir(dir_path) if f.endswith(extension) and exp_name in f and ('t3' in f or 't4' in f)]
        file_name = ''
        file_list.sort()
        # Plot the data
        for file in file_list:
            with open(file, 'rb') as f:
                values_list = pickle.load(f)
                label = 'T3' if 't3' in file else 'T4' if 't4' in file else file
                sns.lineplot(data=values_list, linestyle='--', label=label)
        
    plt.xlabel('Episodes', fontsize=20, labelpad=-2)
    plt.ylabel('Reward', fontsize=20)
    # Setting the plot range values according to the experiment to visualise it better
    box_values = range(-200, 250, 50) if exp_name in 'LunarLand' else range(0,550,50)
    plt.yticks(box_values)
    plt.title('Learning curve for '+ exp_name,fontsize=25)
    plt.legend()
    plt.grid()
    plt.show()



if __name__ == '__main__':
    main()
