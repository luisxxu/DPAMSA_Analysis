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
from parallel_env import ParallelEnvironment
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
    parser.add_argument("--scoring",
                        type=str,
                        default="sp",
                        choices=["sp", "profile", "both"],
                        help=(
                            "Scoring function for reporting alignment quality. "
                            "'sp' (default): Sum-of-Pairs O(L·k²); "
                            "'profile': PSSM-based O(L·k) equivalent; "
                            "'both': print both scores and display the PSSM table."))
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
        if config.n_envs > 1:
            train_dqn_parallel(sequences, args.scoring)
        else:
            train(sequences, args.scoring)
    elif args.algorithm == "a2c":
        if config.n_envs > 1:
            train_a2c_parallel(sequences, args.scoring)
        else:
            train_a2c(sequences, args.scoring)
    elif args.algorithm == "ppo":
        if config.n_envs > 1:
            train_ppo_parallel(sequences, args.scoring)
        else:
            train_ppo(sequences, args.scoring)


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
# Scoring helpers
# ---------------------------------------------------------------------------

# Token names indexed by token ID (pad=0, A=1, T=2, C=3, G=4, -=5, N=6, R=7, W=8, K=9, Y=10)
_TOKEN_NAMES = ['pad', 'A', 'T', 'C', 'G', '-', 'N', 'R', 'W', 'K', 'Y']


def _print_pssm(pssm):
    """Print a PSSM as a human-readable table.

    Rows = alignment positions; columns = token IDs (one column per token name).
    Each cell shows the frequency (0.000 – 1.000) of that token at that position.

    Args:
        pssm: numpy ndarray of shape (L, VOCAB_SIZE) returned by env.build_pssm().
    """
    header = "pos  " + "  ".join(f"{t:>5}" for t in _TOKEN_NAMES)
    print("PSSM (position × token frequency):")
    print(header)
    for i, row in enumerate(pssm):
        print(f"{i:3d}  " + "  ".join(f"{v:5.3f}" for v in row))


def _print_results(env, scoring):
    """Print alignment quality metrics.

    env.padding() must already have been called before this function.

    Args:
        env:     Environment instance with a completed alignment.
        scoring: One of 'sp', 'profile', or 'both'.
                 'sp'      — print Sum-of-Pairs score only (O(L·k²))
                 'profile' — print profile score and PSSM table (O(L·k))
                 'both'    — print SP score, profile score, and PSSM table
    """
    L = len(env.aligned[0])
    print("total length : {}".format(L))
    if scoring in ("sp", "both"):
        print("sp score     : {}".format(env.calc_score()))
    if scoring in ("profile", "both"):
        print("profile score: {}".format(env.calc_profile_score()))
        _print_pssm(env.build_pssm())
    print("exact matched: {}".format(env.calc_exact_matched()))
    print("column score : {}".format(env.calc_exact_matched() / L))
    print("alignment: \n{}".format(env.get_alignment()))


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

def train(sequences, scoring="sp"):
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
    _print_results(env, scoring)
    print("********************************\n")


# ---------------------------------------------------------------------------
# A2C training
# ---------------------------------------------------------------------------

def train_a2c(sequences, scoring="sp"):
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
    _print_results(env, scoring)
    print("********************************\n")


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(sequences, scoring="sp"):
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
    _print_results(env, scoring)
    print("********************************\n")


# ---------------------------------------------------------------------------
# Parallel training helpers
# ---------------------------------------------------------------------------

def _collect_episode_parallel(par_env, forward_fn):
    """Collect one episode from all N envs using batched forward passes.

    forward_fn(active_states) must return (actions, values, log_probs) as
    numpy arrays aligned with active_states.  log_probs may be None for DQN.

    Returns per-env trajectory dicts:
        traj['s']  — states
        traj['a']  — actions
        traj['r']  — rewards
        traj['v']  — value estimates  (None entries for DQN)
        traj['d']  — done flags
        traj['lp'] — log π_old(a|s)  (None entries for DQN)
    """
    n = par_env.n_envs
    states = par_env.reset_all()
    traj   = {k: [[] for _ in range(n)] for k in ('s', 'a', 'r', 'v', 'd', 'lp')}

    while not par_env.all_done():
        active_idx    = [i for i, a in enumerate(par_env.active_mask) if a]
        active_states = [states[i] for i in active_idx]

        # ONE batched forward pass for all active environments
        actions_arr, values_arr, lp_arr = forward_fn(active_states)

        full_actions = [0] * n
        for j, env_i in enumerate(active_idx):
            full_actions[env_i] = int(actions_arr[j])

        results = par_env.step_all(full_actions)

        for j, env_i in enumerate(active_idx):
            reward, next_state, done = results[env_i]
            traj['s'][env_i].append(states[env_i])
            traj['a'][env_i].append(int(actions_arr[j]))
            traj['r'][env_i].append(reward)
            traj['v'][env_i].append(float(values_arr[j]) if values_arr is not None else None)
            traj['d'][env_i].append(float(done))
            traj['lp'][env_i].append(float(lp_arr[j]) if lp_arr is not None else None)
            if done != 0:
                states[env_i] = next_state

    return traj


def _report(env, algorithm_tag, scoring="sp"):
    """Print alignment quality metrics (shared by all parallel training functions).

    Args:
        env:           Environment instance after greedy inference.
        algorithm_tag: Short label printed in the footer (e.g. 'DQN', 'A2C').
        scoring:       Same choices as --scoring: 'sp', 'profile', or 'both'.
    """
    env.padding()
    _print_results(env, scoring)
    print(f"({'parallel ' + algorithm_tag}){'*' * 32}\n")


# ---------------------------------------------------------------------------
# DQN — parallel data collection
# ---------------------------------------------------------------------------

def train_dqn_parallel(sequences, scoring="sp"):
    """DQN with n_envs parallel environments filling the replay buffer faster.

    Each episode collects transitions from all N envs simultaneously using
    agent.select_batch() (one forward pass per step for greedy actions).
    All transitions are pushed to the shared replay buffer, so the buffer
    fills N× faster and the agent begins learning from more diverse
    experience earlier.
    """
    output_parameters()
    n_envs = config.n_envs
    print(f"[DQN parallel] n_envs={n_envs}")

    train_start = time.monotonic()
    print(f"Training on {len(sequences)} sequences (DQN ×{n_envs} envs):")
    for key in sequences:
        print(f"  Sequence {key}")

    seqs      = list(sequences.values())
    par_env   = ParallelEnvironment(seqs, n_envs)
    agent     = DQN(par_env.action_number, par_env.row, par_env.max_len,
                    par_env.max_len * par_env.max_reward)
    p = tqdm(range(config.max_episode))

    for _ in p:
        states = par_env.reset_all()

        while not par_env.all_done():
            active_idx    = [i for i, a in enumerate(par_env.active_mask) if a]
            active_states = [states[i] for i in active_idx]

            # Batched epsilon-greedy: one forward pass for all greedy actions
            batch_actions = agent.select_batch(active_states)

            full_actions = [0] * n_envs
            for j, env_i in enumerate(active_idx):
                full_actions[env_i] = batch_actions[j]

            results = par_env.step_all(full_actions)

            for j, env_i in enumerate(active_idx):
                reward, next_state, done = results[env_i]
                # Push each env's transition into the shared replay buffer
                ns = next_state if next_state is not None else states[env_i]
                agent.replay_memory.push(
                    (states[env_i], ns, full_actions[env_i], reward, done))
                agent.update()
                if done != 0:
                    states[env_i] = next_state

        agent.update_epsilon()

    print(f"Training Complete (DQN ×{n_envs})")
    print(f"Training time: {time.monotonic() - train_start:.2f} seconds")

    # Greedy inference on a single env
    predict_start = time.monotonic()
    single_env = par_env.envs[0]
    state = single_env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = single_env.step(action)
        state = next_state
        if done == 0:
            break
    print(f"Predict time: {time.monotonic() - predict_start:.2f} seconds")
    _report(single_env, "DQN", scoring)


# ---------------------------------------------------------------------------
# A2C — parallel episode collection
# ---------------------------------------------------------------------------

def train_a2c_parallel(sequences, scoring="sp"):
    """A2C with n_envs parallel environments.

    Each episode step batches all active env states into a single
    (N_active, state_dim) tensor for one forward pass through the actor-critic
    network, then calls agent.update_parallel() which concatenates the N
    per-env trajectories before the gradient step.

    Effect: each gradient update sees N× as many transitions as the single-env
    version at almost the same wall-clock cost per step.
    """
    output_parameters_a2c()
    n_envs = config.n_envs
    print(f"[A2C parallel] n_envs={n_envs}")

    train_start = time.monotonic()
    print(f"Training on {len(sequences)} sequences (A2C ×{n_envs} envs):")
    for key in sequences:
        print(f"  Sequence {key}")

    seqs    = list(sequences.values())
    par_env = ParallelEnvironment(seqs, n_envs)
    agent   = ActorCritic(par_env.action_number, par_env.row, par_env.max_len)
    p       = tqdm(range(config.max_episode))

    def forward_fn(active_states):
        x = torch.LongTensor(active_states).to(config.device)
        with torch.no_grad():
            actions_t, _, values_t, _ = agent.net.get_action_and_value(x)
        return (actions_t.cpu().numpy(),
                values_t.cpu().numpy(),
                None)  # A2C does not need log_probs

    for _ in p:
        traj = _collect_episode_parallel(par_env, forward_fn)
        agent.update_parallel(
            traj['s'], traj['a'], traj['r'], traj['v'], traj['d'])

    print(f"Training Complete (A2C ×{n_envs})")
    print(f"Training time: {time.monotonic() - train_start:.2f} seconds")

    # Greedy inference on a single env
    predict_start = time.monotonic()
    single_env = par_env.envs[0]
    state = single_env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = single_env.step(action)
        state = next_state
        if done == 0:
            break
    print(f"Predict time: {time.monotonic() - predict_start:.2f} seconds")
    _report(single_env, "A2C", scoring)


# ---------------------------------------------------------------------------
# PPO — parallel episode collection
# ---------------------------------------------------------------------------

def train_ppo_parallel(sequences, scoring="sp"):
    """PPO with n_envs parallel environments.

    Collects log π_old(a|s) alongside actions so the clipped surrogate can be
    computed correctly during update_parallel().  GAE is computed per trajectory
    then concatenated — each ppo_epochs update sees N× the data of single-env.
    """
    output_parameters_ppo()
    n_envs = config.n_envs
    print(f"[PPO parallel] n_envs={n_envs}")

    train_start = time.monotonic()
    print(f"Training on {len(sequences)} sequences (PPO ×{n_envs} envs):")
    for key in sequences:
        print(f"  Sequence {key}")

    seqs    = list(sequences.values())
    par_env = ParallelEnvironment(seqs, n_envs)
    agent   = PPO(par_env.action_number, par_env.row, par_env.max_len)
    p       = tqdm(range(config.max_episode))

    def forward_fn(active_states):
        x = torch.LongTensor(active_states).to(config.device)
        with torch.no_grad():
            actions_t, log_probs_t, values_t, _ = agent.net.get_action_and_value(x)
        return (actions_t.cpu().numpy(),
                values_t.cpu().numpy(),
                log_probs_t.cpu().numpy())

    for _ in p:
        traj = _collect_episode_parallel(par_env, forward_fn)
        agent.update_parallel(
            traj['s'], traj['a'], traj['lp'], traj['r'], traj['v'], traj['d'])

    print(f"Training Complete (PPO ×{n_envs})")
    print(f"Training time: {time.monotonic() - train_start:.2f} seconds")

    # Greedy inference on a single env
    predict_start = time.monotonic()
    single_env = par_env.envs[0]
    state = single_env.reset()
    while True:
        action = agent.predict(state)
        _, next_state, done = single_env.step(action)
        state = next_state
        if done == 0:
            break
    print(f"Predict time: {time.monotonic() - predict_start:.2f} seconds")
    _report(single_env, "PPO", scoring)


if __name__ == "__main__":
    main()
