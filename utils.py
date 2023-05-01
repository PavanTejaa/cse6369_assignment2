import random
import numpy as np
import torch.cuda


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'


# Util function to apply reward-discounting scheme on a list of instant-reward (from eq 8)
def apply_discount(raw_reward, gamma=0.99):
    # TODO: Compute discounted_rtg_reward (as a list) from raw_reward
    # HINT: Reverse the input list, keep a running-average. Reverse again to get the correct order.
    raw_reward.reverse()
    sum = 0
    discounted_rtg_reward = []
    for reward in raw_reward:
        sum = sum*gamma+reward
        discounted_rtg_reward.append(sum)
    raw_reward.reverse()
    discounted_rtg_reward.reverse()
    # Normalization
    discounted_rtg_reward_as_tensor = torch.stack(discounted_rtg_reward)

    # Calculate the mean and standard deviation of the reward discounts
    mean = torch.mean(discounted_rtg_reward_as_tensor, dim=0)
    std = torch.std(discounted_rtg_reward_as_tensor, dim=0)

    # Normalize the reward discounts
    normalized_tensor_list = [tensor - mean / (std + np.finfo(np.float32).eps) for tensor in discounted_rtg_reward]

    return normalized_tensor_list


# Util function to apply reward-return (cumulative reward) on a list of instant-reward (from eq 6)
def apply_return(raw_reward):
    # Compute r_reward (as a list) from raw_reward
    r_reward = [np.sum(raw_reward)]
    return torch.tensor(r_reward, dtype=torch.float32, device=get_device())