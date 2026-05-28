# Removed the lines of code importing the unknown datasets
import sys
import os
import argparse
from tqdm import tqdm
import config
from env import Environment
from dqn import DQN
from actor_critic import ActorCritic
from ppo import PPO
# The Biopython module is used to load in fasta file datasets
from Bio import SeqIO
import torch
# The time module is used to record the run time for DPAMSA
import time


def main():
    # ---------------------------------------------------------------------------
    # Argument parsing
    # Usage:
    #   python main.py <fasta_file>                        # DQN (default)
    #   python main.py <fasta_file> --algorithm a2c        # A2C
    #   python main.py <fasta_file> --algorithm ppo        # PPO
    #   python main.py <fasta_file> --num_datasets N       # multi-train (DQN)
    # ---------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description="DPAMSA: Deep RL for Multiple Sequence Alignment")
    parser.add_argument("fasta_file",
                        type=str,
                        help="Path to input FASTA file")
    parser.add_argument("--num_datasets",
                        type=int,
                        default=None,
                        help="Split sequences into N sub-datasets for multi-train (DQN only)")
    parser.add_argument("--algorithm",
                        type=str,
                        default="dqn",
                        choices=["dqn", "a2c", "ppo"],
                        help="RL algorithm to use: dqn (default), a2c, or ppo")
    args = parser.parse_args()

    # This line ensures that a GPU node is being used if available
    config.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device = torch.device(config.device_name)

    # Load sequences from the fasta file using the BioPython tools
    sequences = {record.id: str(record.seq)
                 for record in SeqIO.parse(args.fasta_file, "fasta")}

    # This if-then statement ensures that the correct training function is used.
    # For this project, the multi_train function is not used.
    if args.num_datasets is not None:
        multi_train(sequences, args.num_datasets)
    elif args.algorithm == "dqn":
        train(sequences)
    elif args.algorithm == "a2c":
        train_a2c(sequences)
    elif args.algorithm == "ppo":
        train_ppo(sequences)


# ---------------------------------------------------------------------------
# Parameter printers
# ---------------------------------------------------------------------------

def output_parameters():
    """Print DQN hyperparameters."""
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.max_episode))
    print("Batch size: {}".format(config.batch_size))
    print("Replay memory size: {}".format(config.replay_memory_size))
    print("Alpha: {}".format(config.alpha))
    print("Epsilon: {}".format(config.epsilon))
    print("Gamma: {}".format(config.gamma))
    print("Delta: {}".format(config.delta))
    print("Decrement iteration: {}".format(config.decrement_iteration))
    print("Update iteration: {}".format(config.update_iteration))
    print("Device: {}".format(config.device_name))


def output_parameters_a2c():
    """Print A2C hyperparameters."""
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.max_episode))
    print("Gamma: {}".format(config.gamma))
    print("A2C learning rate: {}".format(config.a2c_lr))
    print("A2C value coef: {}".format(config.a2c_value_coef))
    print("A2C entropy coef: {}".format(config.a2c_entropy_coef))
    print("Device: {}".format(config.device_name))


def output_parameters_ppo():
    """Print PPO hyperparameters."""
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.max_episode))
    print("Gamma: {}".format(config.gamma))
    print("PPO learning rate: {}".format(config.ppo_lr))
    print("PPO clip epsilon: {}".format(config.ppo_clip_eps))
    print("PPO epochs: {}".format(config.ppo_epochs))
    print("PPO value coef: {}".format(config.ppo_value_coef))
    print("PPO entropy coef: {}".format(config.ppo_entropy_coef))
    print("GAE lambda: {}".format(config.gae_lambda))
    print("Device: {}".format(config.device_name))


# ---------------------------------------------------------------------------
# Multi-train (DQN only — not used in this project)
# ---------------------------------------------------------------------------

# Although not used in this project, this function was altered to ensure that
# the fasta files are loaded in properly
def multi_train(sequences, num_datasets):
    output_parameters()
    print("Dataset number: {}".format(num_datasets))

    report_file_name = os.path.join(config.report_path, "multi_train.rpt")

    with open(report_file_name, 'w') as _:
        _.truncate()

    # Split sequences into datasets
    seq_per_dataset = len(sequences) // num_datasets
    datasets = [sequences[i:i + seq_per_dataset]
                for i in range(0, len(sequences), seq_per_dataset)]

    # Train on each dataset
    for index, seqs in enumerate(datasets):
        env = Environment(seqs)
        agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
        p = tqdm(range(config.max_episode))
        p.set_description(f"Dataset {index + 1}")

        for _ in p:
            state = env.reset()
            while True:
                action = agent.select(state)
                reward, next_state, done = env.step(action)
                agent.replay_memory.push((state, next_state, action, reward, done))
                agent.update()
                if done == 0:
                    break
                state = next_state
            agent.update_epsilon()

        state = env.reset()

        while True:
            action = agent.predict(state)
            _, next_state, done = env.step(action)
            state = next_state
            if 0 == done:
                break

        env.padding()
        report = "{}\n{}\n{}\n{}\n{}\n{}\n{}\n\n".format(
            "NO: {}".format(name),
            "AL: {}".format(len(env.aligned[0])),
            "SP: {}".format(env.calc_score()),
            "EM: {}".format(env.calc_exact_matched()),
            "CS: {}".format(env.calc_exact_matched() / len(env.aligned[0])),
            "QTY: {}".format(len(env.aligned)),
            "#\n{}".format(env.get_alignment()))

        with open(os.path.join(config.report_path, "{}.rpt".format(tag)), 'a+') as report_file:
            report_file.write(report)


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------

def train(sequences):
    output_parameters()

# Commented out the lines below as they relate to the unspecified data format.
#    assert hasattr(dataset, "dataset_{}".format(index)), "No such data called {}".format("dataset_{}".format(index))
#    data = getattr(dataset, "dataset_{}".format(index))
#    print("{}: dataset_{}: {}".format(dataset.file_name, index, data))
    # Set the start time to record the run time
    train_start_time = time.monotonic()
    # These print statements confirm the sequences that are being aligned
    print(f"Training on {len(sequences)} sequences:")
    for key in sequences:
        print(f"Sequence {key}")
    env = Environment(list(sequences.values()))
    agent = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
    p = tqdm(range(config.max_episode))

    for _ in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            agent.replay_memory.push((state, next_state, action, reward, done))
            agent.update()
            if done == 0:
                break
            state = next_state
        agent.update_epsilon()
    # The end time for training is recorded for run time calculation
    train_end_time = time.monotonic()
    # Print statement for checkpoint confirmation
    print("Training Complete")
    # Run time calculation
    train_time = train_end_time - train_start_time
    # Print training time formatted to 2 decimal places
    print(f"Training time: {train_time:.2f} seconds")

    # Predicting the alignment based off the training
    # Record start time for run time calculation
    predict_start_time = time.monotonic()
    state = env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = env.step(action)
        state = next_state
        if 0 == done:
            break
    # Record end time for run time calculation
    predict_end_time = time.monotonic()
    # Print statement for checkpoint confirmation
    print("Prediction Complete")
    # Calculate the prediction time
    predict_time = predict_end_time - predict_start_time
    # Print predicting time formatted to 2 decimal places
    print(f"Predict time: {predict_time:.2f} seconds")

    env.padding()
#    print("**********dataset: {} **********\n".format(data))
    print("total length : {}".format(len(env.aligned[0])))
    print("sp score     : {}".format(env.calc_score()))
    print("exact matched: {}".format(env.calc_exact_matched()))
    print("column score : {}".format(env.calc_exact_matched() / len(env.aligned[0])))
    print("alignment: \n{}".format(env.get_alignment()))
    print("********************************\n")


# ---------------------------------------------------------------------------
# A2C training
# ---------------------------------------------------------------------------

def train_a2c(sequences):
    """Train an Advantage Actor-Critic agent on the provided sequences.

    Loop structure:
        for each episode:
            collect full episode → agent.select() + agent.record_transition()
            update policy       → agent.update()   (single gradient step)
        run greedy inference    → agent.predict()
    """
    output_parameters_a2c()
    train_start_time = time.monotonic()

    print(f"Training on {len(sequences)} sequences (A2C):")
    for key in sequences:
        print(f"Sequence {key}")

    env   = Environment(list(sequences.values()))
    agent = ActorCritic(env.action_number, env.row, env.max_len)
    p     = tqdm(range(config.max_episode))

    for _ in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            # Record the reward and done signal for the step just taken
            agent.record_transition(reward, float(done))
            if done == 0:
                break
            state = next_state
        # Update policy on the completed episode trajectory
        agent.update()

    train_end_time = time.monotonic()
    print("Training Complete (A2C)")
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")

    # Greedy inference
    predict_start_time = time.monotonic()
    state = env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = env.step(action)
        state = next_state
        if 0 == done:
            break
    predict_end_time = time.monotonic()
    print("Prediction Complete (A2C)")
    print(f"Predict time: {predict_end_time - predict_start_time:.2f} seconds")

    env.padding()
    print("total length : {}".format(len(env.aligned[0])))
    print("sp score     : {}".format(env.calc_score()))
    print("exact matched: {}".format(env.calc_exact_matched()))
    print("column score : {}".format(env.calc_exact_matched() / len(env.aligned[0])))
    print("alignment: \n{}".format(env.get_alignment()))
    print("********************************\n")


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(sequences):
    """Train a PPO-Clip agent on the provided sequences.

    Loop structure:
        for each episode:
            collect full episode  → agent.select() + agent.record_transition()
            K-epoch PPO update    → agent.update()
                (computes GAE advantages, runs ppo_epochs clipped updates)
        run greedy inference     → agent.predict()
    """
    output_parameters_ppo()
    train_start_time = time.monotonic()

    print(f"Training on {len(sequences)} sequences (PPO):")
    for key in sequences:
        print(f"Sequence {key}")

    env   = Environment(list(sequences.values()))
    agent = PPO(env.action_number, env.row, env.max_len)
    p     = tqdm(range(config.max_episode))

    for _ in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            # Record the reward and done signal for the step just taken
            agent.record_transition(reward, float(done))
            if done == 0:
                break
            state = next_state
        # Run ppo_epochs gradient steps on the collected episode rollout
        agent.update()

    train_end_time = time.monotonic()
    print("Training Complete (PPO)")
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")

    # Greedy inference
    predict_start_time = time.monotonic()
    state = env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = env.step(action)
        state = next_state
        if 0 == done:
            break
    predict_end_time = time.monotonic()
    print("Prediction Complete (PPO)")
    print(f"Predict time: {predict_end_time - predict_start_time:.2f} seconds")

    env.padding()
    print("total length : {}".format(len(env.aligned[0])))
    print("sp score     : {}".format(env.calc_score()))
    print("exact matched: {}".format(env.calc_exact_matched()))
    print("column score : {}".format(env.calc_exact_matched() / len(env.aligned[0])))
    print("alignment: \n{}".format(env.get_alignment()))
    print("********************************\n")


if __name__ == "__main__":
    main()
