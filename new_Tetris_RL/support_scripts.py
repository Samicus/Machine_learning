
import collections
import matplotlib.pyplot as plt
import numpy as np
import torch

def calculate_q(network, state, device):
    """
    Calculate Q-values for the given state. If you want to calculate Q to take action,
    use offline model.
    :param network: The DDQL agent object
    :param state: State for which to calculate Q-values
    :param device: Device to train on
    :return: Q values
    """
    tensor_state = torch.tensor(state, dtype=torch.float32, device=device)

    # Calculate Q-values
    q_values = network.forward(tensor_state).detach().cpu().numpy()

    return q_values


def calculate_q_targets(
    q1_batch, q2_batch, r_batch, nonterminal_batch, device, gamma=0.99
):
    """
    Calculates the Q target used for the loss
    : param q1_batch: Batch of Q(s', a) from online network. FloatTensor, shape (N, num actions)
    : param q2_batch: Batch of Q(s', a) from target network. FloatTensor, shape (N, num actions)
    : param r_batch: Batch of rewards. FloatTensor, shape (N,)
    : param nonterminal_batch: Batch of booleans, with False elements if state s' is terminal and True otherwise.
                                BoolTensor, shape (N,)
    : param gamma: Discount factor, float.
    : return: Q target. FloatTensor, shape (N,)
    """

    Y = torch.tensor(
        np.zeros(len(nonterminal_batch)), dtype=torch.float32, device=device
    )

    for i in range(len(Y)):
        if nonterminal_batch[i]:
            Y[i] = (
                r_batch[i]
                + gamma * q2_batch[i, np.argmax(q1_batch[i, :].detach().cpu().numpy())]
            )
        else:
            Y[i] = r_batch[i]

    return Y


def sample_batch_and_calculate_loss(agent, replay_buffer, batch_size, gamma, device):
    """
    Sample mini-batch from replay buffer, and compute the mini-batch loss
    Inputs:
        ddqn          - DDQN model. An object holding the online / offline Q-networks, and some related methods.
        replay_buffer - Replay buffer object (from which smaples will be drawn)
        batch_size    - Batch size
        gamma         - Discount factor
    Returns:
        Mini-batch loss, on which .backward() will be called to compute gradient.
    """

    (
        curr_state,
        curr_action,
        reward,
        next_state,
        nonterminal,
    ) = replay_buffer.sample_minibatch_tensor(batch_size)

    # Calculate Q values at next state for offline and online model
    q_online_next = agent.online_model.forward(next_state)

    with torch.no_grad():
        q_offline_next = agent.offline_model.forward(next_state)

    # Calculate Q targets
    q_target = calculate_q_targets(
        q_online_next, q_offline_next, reward, nonterminal, device, gamma
    )

    # Calculate Q values at current states
    q_online_curr = agent.online_model.forward(curr_state)

    # Compute the loss
    loss = agent.calc_loss(q_online_curr, q_target, curr_action)

    del q_target, q_online_curr

    return loss


def calculate_loss(
    agent, gamma, device, curr_state, curr_action, reward, next_state, nonterminal
):
    # Calculate Q values at next state for offline and online model
    q_online_next = agent.online_model.forward(next_state)

    with torch.no_grad():
        q_offline_next = agent.offline_model.forward(next_state)

    # Calculate Q targets
    q_target = calculate_q_targets(
        q_online_next, q_offline_next, reward, nonterminal, device, gamma
    )

    # Calculate Q values at current states
    q_online_curr = agent.online_model.forward(curr_state)

    # Compute the loss
    loss = agent.calc_loss(q_online_curr, q_target, curr_action)

    del q_target, q_online_curr

    return loss

def plot_rewards(reward_tots):
    print("a")
    plt.figure()
    print("b")
    plt.plot(reward_tots)
    print("c")
    plt.plot(range(len(reward_tots)), smooth(reward_tots, 20))
    print("d")
    plt.xlabel("Episodes")
    print("e")
    plt.ylabel("Reward")
    print("f")
    plt.show()
    print("g")
def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

def binatodeci(binary):
    deci = sum(val*(2**idx) for idx, val in enumerate(reversed(binary)))
    return int(deci)

