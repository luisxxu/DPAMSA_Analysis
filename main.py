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
from acer import ACER
from parallel_env import ParallelEnvironment
import benchmark_compare as bc
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
                        choices=["dqn", "a2c", "ppo", "acer"],
                        help="RL algorithm to use: dqn (default), a2c, ppo, or acer")
    parser.add_argument("--episodes",
                        type=int,
                        default=None,
                        help="Number of training episodes (overrides config.max_episode)")
    parser.add_argument("--save",
                        type=str,
                        default=None,
                        metavar="NAME",
                        help="Save model weights to weights/NAME.pth after training")
    parser.add_argument("--load",
                        type=str,
                        default=None,
                        metavar="NAME",
                        help="Load model weights from weights/NAME.pth before training "
                             "(resume a previous run)")
    parser.add_argument("--checkpoint",
                        type=int,
                        default=0,
                        metavar="N",
                        help="Save a checkpoint every N episodes during training "
                             "(requires --save; filenames: weights/NAME_epN.pth)")
    parser.add_argument("--scoring",
                        type=str,
                        default="sp",
                        choices=["sp", "profile", "both"],
                        help=(
                            "Scoring function for reporting alignment quality. "
                            "'sp' (default): Sum-of-Pairs O(L·k²); "
                            "'profile': PSSM-based O(L·k) equivalent; "
                            "'both': print both scores and display the PSSM table."))

    # ── ACER hyperparameter overrides ─────────────────────────────────────────
    # These let you tune ACER from the command line without editing config.py.
    # Useful when scaling to harder datasets (more sequences / longer sequences)
    # where the default 3×30 bp settings are too aggressive or too conservative.
    acer_group = parser.add_argument_group(
        "ACER hyperparameters",
        "Override config.py defaults for ACER (ignored for DQN / A2C / PPO).")
    acer_group.add_argument("--acer-lr", type=float, default=None,
                            metavar="LR",
                            help="Adam learning rate (default: %(default)s → "
                                 f"config.acer_lr={config.acer_lr})")
    acer_group.add_argument("--acer-entropy", type=float, default=None,
                            metavar="COEF",
                            help="Entropy bonus coefficient.  Raise this for "
                                 "larger action spaces to prevent premature "
                                 "policy collapse  (default → "
                                 f"config.acer_entropy_coef={config.acer_entropy_coef})")
    acer_group.add_argument("--acer-replay-ratio", type=int, default=None,
                            metavar="N",
                            help="Off-policy updates per on-policy step.  "
                                 "Reduce for larger action spaces where stale "
                                 "buffer episodes cause instability "
                                 f"(default → config.acer_replay_ratio={config.acer_replay_ratio})")
    acer_group.add_argument("--acer-cbar", type=float, default=None,
                            metavar="C",
                            help="IS truncation threshold c̄ "
                                 f"(default → config.acer_c_bar={config.acer_c_bar})")
    acer_group.add_argument("--acer-trust-delta", type=float, default=None,
                            metavar="D",
                            help="KL penalty coefficient for trust-region "
                                 f"(default → config.acer_trust_region_delta="
                                 f"{config.acer_trust_region_delta})")
    acer_group.add_argument("--acer-entropy-end", type=float, default=None,
                            metavar="COEF",
                            help="If set, the entropy coefficient is annealed "
                                 "exponentially from --acer-entropy down to this "
                                 "value over the full episode budget.  Encourages "
                                 "broad exploration early and tight exploitation "
                                 "late.  Example: --acer-entropy 1.0 "
                                 "--acer-entropy-end 0.01")
    acer_group.add_argument("--acer-inference-rollouts", type=int, default=1,
                            metavar="N",
                            help="Number of stochastic rollouts to run at "
                                 "inference time; the alignment with the highest "
                                 "SP score is used.  N=1 = single greedy rollout "
                                 "(default).  N≥2 provides a free improvement at "
                                 "the cost of ~N× longer inference.")
    acer_group.add_argument("--pretrain", action="store_true",
                            help="Before RL training, pretrain the ACER actor "
                                 "via behavioural cloning on the MAFFT alignment "
                                 "of the input FASTA.  Gives the network a warm "
                                 "start near classical-tool quality so RL refines "
                                 "from a good policy rather than random init.  "
                                 "Requires MAFFT to be available (MAFFT_BIN or "
                                 "system PATH).")
    acer_group.add_argument("--pretrain-epochs", type=int, default=100,
                            metavar="N",
                            help="Number of behavioural-cloning epochs when "
                                 "--pretrain is set (default: 100).  Each epoch "
                                 "is one gradient step over the full expert "
                                 "trajectory (~alignment_length steps).")

    # ── Benchmark comparison ───────────────────────────────────────────────────
    bench_group = parser.add_argument_group(
        "Benchmark comparison",
        "Compare the trained agent against pre-computed classical tool scores.")
    bench_group.add_argument("--results-csv", type=str, default=None,
                             metavar="PATH",
                             help="Append SP scores + benchmark data to this CSV "
                                  "(created if absent). Used to accumulate results "
                                  "across multiple runs for figure generation.")
    bench_group.add_argument("--figures-dir", type=str, default=None,
                             metavar="DIR",
                             help="Directory for benchmark comparison figures. "
                                  "Figures are regenerated from --results-csv after "
                                  "each run so they stay current during a batch loop.")
    bench_group.add_argument("--no-compare", action="store_true",
                             help="Skip benchmark comparison even when benchmark "
                                  "data exists for the dataset.")

    # ── Early stopping / convergence ──────────────────────────────────────────
    conv_group = parser.add_argument_group(
        "Early stopping",
        "Stop training early when the SP score converges or reaches a target, "
        "instead of always running the full --episodes count.")
    conv_group.add_argument("--eval-interval", type=int, default=100, metavar="N",
                            help="Run a greedy evaluation pass every N episodes to "
                                 "track the current SP score (default: 100; 0 = off)")
    conv_group.add_argument("--patience", type=int, default=0, metavar="N",
                            help="Stop after N consecutive evaluations with no SP "
                                 "improvement.  0 = never stop early (default). "
                                 "Example: --patience 5 --eval-interval 100 stops "
                                 "after 500 stagnant episodes.")
    conv_group.add_argument("--min-delta", type=float, default=0.0, metavar="D",
                            help="Minimum SP score gain to count as an improvement "
                                 "and reset patience (default: 0 = any gain counts)")
    conv_group.add_argument("--target-rank", type=int, default=None, metavar="R",
                            help="Stop as soon as ACER achieves rank R or better "
                                 "against the benchmark tools.  Requires --results-csv "
                                 "so the tool can look up classical scores.  "
                                 "E.g. --target-rank 1 stops the moment ACER beats "
                                 "every classical tool; --target-rank 2 stops when "
                                 "only one classical tool still beats ACER.")

    args = parser.parse_args()

    # This line ensures that a GPU node is being used if available
    config.device_name = "cuda:0" if torch.cuda.is_available() else "cpu"
    config.device = torch.device(config.device_name)

    if args.episodes is not None:
        config.max_episode = args.episodes

    # Apply ACER overrides
    if args.acer_lr            is not None: config.acer_lr                 = args.acer_lr
    if args.acer_entropy       is not None: config.acer_entropy_coef       = args.acer_entropy
    if args.acer_replay_ratio  is not None: config.acer_replay_ratio       = args.acer_replay_ratio
    if args.acer_cbar          is not None: config.acer_c_bar              = args.acer_cbar
    if args.acer_trust_delta   is not None: config.acer_trust_region_delta = args.acer_trust_delta
    # acer_entropy_end and acer_inference_rollouts are passed directly to train_acer

    # Load sequences from the fasta file using the BioPython tools
    sequences = {record.id: str(record.seq)
                 for record in SeqIO.parse(args.fasta_file, "fasta")}

    # This if-then statement ensures that the correct training function is used.
    # For this project, the multi_train function is not used.
    ckpt_args = dict(
        save=args.save, load=args.load, checkpoint=args.checkpoint,
        fasta_path=args.fasta_file,
        results_csv=args.results_csv,
        figures_dir=args.figures_dir,
        no_compare=args.no_compare,
        eval_interval=args.eval_interval,
        patience=args.patience,
        min_delta=args.min_delta,
        target_rank=args.target_rank,
        entropy_end=args.acer_entropy_end,
        inference_rollouts=args.acer_inference_rollouts,
        pretrain=args.pretrain,
        pretrain_epochs=args.pretrain_epochs,
    )

    if args.num_datasets is not None:
        multi_train(sequences, args.num_datasets)
    elif args.algorithm == "dqn":
        if config.n_envs > 1:
            train_dqn_parallel(sequences, args.scoring)
        else:
            train(sequences, args.scoring, **ckpt_args)
    elif args.algorithm == "a2c":
        if config.n_envs > 1:
            train_a2c_parallel(sequences, args.scoring)
        else:
            train_a2c(sequences, args.scoring, **ckpt_args)
    elif args.algorithm == "ppo":
        if config.n_envs > 1:
            train_ppo_parallel(sequences, args.scoring)
        else:
            train_ppo(sequences, args.scoring, **ckpt_args)
    elif args.algorithm == "acer":
        train_acer(sequences, args.scoring, **ckpt_args)


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
# Weight persistence helpers
# ---------------------------------------------------------------------------

_WEIGHT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")


def _weight_dir() -> str:
    """Return (and create if necessary) the local weights/ directory."""
    os.makedirs(_WEIGHT_DIR, exist_ok=True)
    return _WEIGHT_DIR


def _maybe_load(agent, name):
    """Load weights from weights/NAME.pth if --load was given."""
    if name:
        agent.load(name, path=_weight_dir())


def _maybe_save(agent, name):
    """Save weights to weights/NAME.pth if --save was given."""
    if name:
        agent.save(name, path=_weight_dir())


def _maybe_checkpoint(agent, name, interval, episode):
    """Save a checkpoint if --checkpoint N and this is a checkpoint episode."""
    if name and interval > 0 and (episode + 1) % interval == 0:
        agent.save(f"{name}_ep{episode + 1}", path=_weight_dir())
        print(f"  [checkpoint] weights/{name}_ep{episode + 1}.pth")


def _bench_classical_scores(fasta_path: str) -> list[float]:
    """Return a sorted (best→worst) list of available classical SP scores.

    Used by _check_early_stop to evaluate target-rank conditions without
    re-reading the CSV on every evaluation interval.  Returns [] if the
    benchmark file cannot be found or parsed.
    """
    try:
        record = bc.lookup(fasta_path, agent_score=0)
        scores = [v for k, v in record.items()
                  if k.startswith("SP_") and k != "SP_ACER" and v is not None]
        return sorted(scores, reverse=True)   # best (highest) first
    except Exception:
        return []


def _check_early_stop(agent, seqs_list: list, ep: int,
                      best_score: float, patience_counter: int,
                      patience: int, min_delta: float,
                      classical_scores: list, target_rank,
                      save: str | None, pbar=None,
                      n_history: int = 0) -> tuple:
    """Run one greedy inference pass and test all early-stopping conditions.

    Returns
    -------
    eval_score      : int   — SP score of the greedy policy at this checkpoint
    best_score      : float — updated best score seen so far
    patience_counter: int   — updated patience counter
    stop            : bool  — True if training should stop now
    """
    # Fresh environment so training state is not disturbed.
    # n_history must match the value used when the agent was constructed so
    # the state vector dimensions are compatible with the network weights.
    eval_env = Environment(seqs_list, n_history=n_history)
    state    = eval_env.reset()
    while True:
        action = agent.predict(state)
        _, ns, done = eval_env.step(action)
        state = ns
        if done == 0:
            break
    eval_env.padding()
    eval_score = eval_env.calc_score()

    improved = eval_score > best_score + min_delta
    new_best = eval_score if improved else best_score
    new_pat  = 0 if improved else patience_counter + 1

    # Always keep a "best so far" checkpoint so a crash never loses the peak
    if improved and save:
        agent.save(f"{save}_best", path=_weight_dir())

    # Update tqdm postfix
    if pbar is not None:
        postfix = {"SP": eval_score, "best": int(new_best)}
        if patience > 0:
            postfix["pat"] = f"{new_pat}/{patience}"
        pbar.set_postfix(postfix)

    # ── Patience check ────────────────────────────────────────────────────────
    if patience > 0 and new_pat >= patience:
        print(f"\n  [early stop] No improvement for {new_pat} eval intervals "
              f"(ep {ep + 1}).  Best SP = {int(new_best)}.")
        return eval_score, new_best, new_pat, True

    # ── Target-rank check ─────────────────────────────────────────────────────
    if target_rank is not None and classical_scores:
        # rank = number of classical tools with a better score + 1
        rank = sum(1 for s in classical_scores if s > eval_score) + 1
        if rank <= target_rank:
            beaten = [s for s in classical_scores if eval_score >= s]
            print(f"\n  [early stop] ACER rank {rank} ≤ target {target_rank} "
                  f"at ep {ep + 1}  (SP = {eval_score}, "
                  f"beats {len(beaten)}/{len(classical_scores)} classical tools).")
            return eval_score, new_best, new_pat, True

    return eval_score, new_best, new_pat, False


def _run_mafft(fasta_path: str, seqs_list: list,
               return_alignment: bool = False):
    """Run MAFFT on *fasta_path* and return its SP score.

    The aligned output is scored with the same gap/match/mismatch penalties
    used during RL training so the result is directly comparable.

    Args:
        return_alignment: when True, return (score, aligned_seqs) instead of
            just score.  Used by the imitation-learning pretrain path.

    Returns:
        score (int | None) normally, or (score, aligned_seqs) when
        return_alignment=True.  Both values are None on failure.
    """
    import subprocess
    import io as _io

    _FAIL = (None, None) if return_alignment else None

    mafft_cmd = os.environ.get("MAFFT_BIN", "mafft")

    # Build child env: strip any stale MAFFT_BINARIES, then re-point it at the
    # libexec/ directory that lives *next to our downloaded binary*.
    # Simply unsetting is not enough — DataHub's conda activation scripts
    # re-inject MAFFT_BINARIES inside child processes, overriding the unset.
    child_env = {k: v for k, v in os.environ.items() if k != "MAFFT_BINARIES"}
    if os.path.isabs(mafft_cmd):
        _libexec = os.path.join(os.path.dirname(mafft_cmd), "libexec")
        if os.path.isdir(_libexec):
            child_env["MAFFT_BINARIES"] = _libexec

    try:
        proc = subprocess.run(
            # --inputorder: output sequences in the same order as the input
            # FASTA so the action-bitmask decoding stays in sync with seqs_list.
            [mafft_cmd, "--auto", "--quiet", "--inputorder", fasta_path],
            capture_output=True, text=True, timeout=300,
            env=child_env,
        )
    except FileNotFoundError:
        print(f"  [MAFFT] not found ({mafft_cmd}) — skipping MAFFT comparison")
        return _FAIL
    except subprocess.TimeoutExpired:
        print("  [MAFFT] timed out after 300 s")
        return _FAIL

    if proc.returncode != 0:
        print(f"  [MAFFT] error (exit {proc.returncode}): {proc.stderr[:200]}")
        return _FAIL

    from Bio import SeqIO as _SeqIO
    aligned_seqs = [str(r.seq) for r in _SeqIO.parse(_io.StringIO(proc.stdout), "fasta")]

    if len(aligned_seqs) != len(seqs_list):
        print(f"  [MAFFT] sequence count mismatch "
              f"({len(aligned_seqs)} output vs {len(seqs_list)} input)")
        return _FAIL

    eval_env = Environment(seqs_list)
    eval_env.set_alignment(aligned_seqs)
    score = eval_env.calc_score()
    print(f"  [MAFFT] SP score = {score}")

    if return_alignment:
        return score, aligned_seqs
    return score


def _decode_alignment_to_actions(env, aligned_seqs: list) -> list:
    """Convert a reference MSA into expert (state, action) pairs for BC.

    Each column of the alignment maps to one environment action:
        bit i = 1  →  gap inserted into sequence i
        bit i = 0  →  sequence i advances (emits its next character)

    The environment is reset at the start and stepped column-by-column.
    Collection stops when done=0 (all sequences exhausted) or the alignment
    is fully consumed.  All-gap columns (action = 2^row − 1) are skipped
    because they are not part of the valid action space.

    Returns:
        list of (state, action) tuples for behavioural cloning.
    """
    n_seqs  = len(aligned_seqs)
    aln_len = len(aligned_seqs[0])
    state   = env.reset()
    expert  = []

    for col in range(aln_len):
        action = 0
        for i in range(n_seqs):
            if aligned_seqs[i][col] == '-':
                action |= (1 << i)

        # Skip all-gap columns (undefined in the action space)
        if action == (1 << n_seqs) - 1:
            continue

        expert.append((state, action))
        _, next_state, done = env.step(action)
        state = next_state
        if done == 0:
            break

    return expert


def _pretrain_acer(agent, expert_data: list, n_epochs: int) -> None:
    """Pretrain the ACER actor via behavioural cloning on expert (state, action) pairs.

    Minimises cross-entropy between the policy distribution and the expert
    action at each step.  Only the actor head (and shared encoder) are
    updated; the Q-head is left for the subsequent RL phase.  The EMA
    average-policy network is synced to the pretrained weights so the trust
    region starts centred on the imitation policy.

    Args:
        agent:       ACER instance (must already be on the correct device).
        expert_data: list of (state, action) tuples from _decode_alignment_to_actions.
        n_epochs:    number of full passes over expert_data.
    """
    import torch.nn.functional as _F

    if not expert_data:
        print("  [Pretrain] No expert data available — skipping")
        return

    print(f"  [Pretrain] Behavioural cloning: "
          f"{len(expert_data)} expert steps × {n_epochs} epochs ...")

    states  = torch.LongTensor([s for s, _ in expert_data]).to(config.device)
    actions = torch.LongTensor([a for _, a in expert_data]).to(config.device)

    report_every = max(1, n_epochs // 5)
    agent.net.train()

    for epoch in range(n_epochs):
        with torch.amp.autocast("cuda", enabled=agent._use_amp):
            logits, _, _ = agent.net(states)
            loss = _F.cross_entropy(logits, actions)

        agent.optimizer.zero_grad()
        agent.scaler.scale(loss).backward()
        agent.scaler.unscale_(agent.optimizer)
        torch.nn.utils.clip_grad_norm_(agent.net.parameters(), max_norm=10.0)
        agent.scaler.step(agent.optimizer)
        agent.scaler.update()

        if (epoch + 1) % report_every == 0:
            print(f"  [Pretrain] epoch {epoch + 1:4d}/{n_epochs}  "
                  f"BC-loss={loss.item():.4f}")

    # Sync EMA average network so the trust region starts at the pretrained policy
    with torch.no_grad():
        agent.avg_net.load_state_dict(agent.net.state_dict())
    print("  [Pretrain] Complete — avg_net synced to pretrained weights")


def _run_benchmark_comparison(fasta_path, sp_score, episodes, time_s,
                               results_csv, figures_dir, no_compare,
                               mafft_score=None):
    """Look up benchmark data, print comparison, update CSV and figures.

    Silently skips if:
      • no_compare is True
      • the FASTA path does not match a known dataset (e.g. a custom file)
      • no benchmark CSV exists for the dataset
    """
    if no_compare:
        return
    try:
        record = bc.lookup(fasta_path, sp_score,
                           episodes=episodes, time_s=time_s,
                           mafft_score=mafft_score)
        bc.print_comparison(record)
        if results_csv:
            bc.append_results(record, results_csv)
            print(f"  [results] appended → {results_csv}")
            if figures_dir:
                paths = bc.generate_figures(results_csv, figures_dir)
                for p in paths:
                    print(f"  [figure]  saved → {p}")
    except (ValueError, KeyError, FileNotFoundError) as exc:
        # Benchmark data unavailable for this file — skip silently
        print(f"  [benchmark] skipped: {exc}")


# ---------------------------------------------------------------------------
# DQN training
# ---------------------------------------------------------------------------

def train(sequences, scoring="sp", save=None, load=None, checkpoint=0,
          fasta_path=None, results_csv=None, figures_dir=None, no_compare=False,
          eval_interval=100, patience=0, min_delta=0.0, target_rank=None):
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
    seqs_list = list(sequences.values())
    env       = Environment(seqs_list)
    agent     = DQN(env.action_number, env.row, env.max_len, env.max_len * env.max_reward)
    _maybe_load(agent, load)
    classical = _bench_classical_scores(fasta_path) if fasta_path else []
    p = tqdm(range(config.max_episode))

    best_score, patience_counter = -float("inf"), 0
    ep = 0
    for ep in p:
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
        _maybe_checkpoint(agent, save, checkpoint, ep)

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            _, best_score, patience_counter, stop = _check_early_stop(
                agent, seqs_list, ep, best_score, patience_counter,
                patience, min_delta, classical, target_rank, save, p)
            if stop:
                break

    episodes_run = ep + 1
    # The end time for training is recorded for run time calculation
    train_end_time = time.monotonic()
    _maybe_save(agent, save)
    # Print statement for checkpoint confirmation
    print(f"Training Complete — {episodes_run} episodes")
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
    sp_score = env.calc_score()
    _print_results(env, scoring)
    print("********************************\n")
    if fasta_path:
        mafft_score = _run_mafft(fasta_path, seqs_list)
        _run_benchmark_comparison(fasta_path, sp_score,
                                  episodes_run, train_time,
                                  results_csv, figures_dir, no_compare,
                                  mafft_score=mafft_score)


# ---------------------------------------------------------------------------
# A2C training
# ---------------------------------------------------------------------------

def train_a2c(sequences, scoring="sp", save=None, load=None, checkpoint=0,
              fasta_path=None, results_csv=None, figures_dir=None, no_compare=False,
              eval_interval=100, patience=0, min_delta=0.0, target_rank=None):
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

    seqs_list = list(sequences.values())
    env       = Environment(seqs_list)
    agent     = ActorCritic(env.action_number, env.row, env.max_len)
    _maybe_load(agent, load)
    classical = _bench_classical_scores(fasta_path) if fasta_path else []
    p         = tqdm(range(config.max_episode))

    best_score, patience_counter = -float("inf"), 0
    ep = 0
    for ep in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            agent.record_transition(reward, float(done))
            if done == 0:
                break
            state = next_state
        agent.update()
        _maybe_checkpoint(agent, save, checkpoint, ep)

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            _, best_score, patience_counter, stop = _check_early_stop(
                agent, seqs_list, ep, best_score, patience_counter,
                patience, min_delta, classical, target_rank, save, p)
            if stop:
                break

    episodes_run = ep + 1
    train_end_time = time.monotonic()
    _maybe_save(agent, save)
    print(f"Training Complete (A2C) — {episodes_run} episodes")
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
    sp_score = env.calc_score()
    _print_results(env, scoring)
    print("********************************\n")
    if fasta_path:
        mafft_score = _run_mafft(fasta_path, seqs_list)
        _run_benchmark_comparison(fasta_path, sp_score,
                                  episodes_run,
                                  train_end_time - train_start_time,
                                  results_csv, figures_dir, no_compare,
                                  mafft_score=mafft_score)


# ---------------------------------------------------------------------------
# PPO training
# ---------------------------------------------------------------------------

def train_ppo(sequences, scoring="sp", save=None, load=None, checkpoint=0,
              fasta_path=None, results_csv=None, figures_dir=None, no_compare=False,
              eval_interval=100, patience=0, min_delta=0.0, target_rank=None):
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

    seqs_list = list(sequences.values())
    env       = Environment(seqs_list)
    agent     = PPO(env.action_number, env.row, env.max_len)
    _maybe_load(agent, load)
    classical = _bench_classical_scores(fasta_path) if fasta_path else []
    p         = tqdm(range(config.max_episode))

    best_score, patience_counter = -float("inf"), 0
    ep = 0
    for ep in p:
        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            agent.record_transition(reward, float(done))
            if done == 0:
                break
            state = next_state
        agent.update()
        _maybe_checkpoint(agent, save, checkpoint, ep)

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            _, best_score, patience_counter, stop = _check_early_stop(
                agent, seqs_list, ep, best_score, patience_counter,
                patience, min_delta, classical, target_rank, save, p)
            if stop:
                break

    episodes_run = ep + 1
    train_end_time = time.monotonic()
    _maybe_save(agent, save)
    print(f"Training Complete (PPO) — {episodes_run} episodes")
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
    sp_score = env.calc_score()
    _print_results(env, scoring)
    print("********************************\n")
    if fasta_path:
        mafft_score = _run_mafft(fasta_path, seqs_list)
        _run_benchmark_comparison(fasta_path, sp_score,
                                  episodes_run,
                                  train_end_time - train_start_time,
                                  results_csv, figures_dir, no_compare,
                                  mafft_score=mafft_score)


# ---------------------------------------------------------------------------
# ACER training
# ---------------------------------------------------------------------------

def output_parameters_acer():
    """Print ACER hyperparameters."""
    print("Gap penalty: {}".format(config.GAP_PENALTY))
    print("Mismatch penalty: {}".format(config.MISMATCH_PENALTY))
    print("Match reward: {}".format(config.MATCH_REWARD))
    print("Episode: {}".format(config.max_episode))
    print("Gamma: {}".format(config.gamma))
    print("ACER learning rate: {}".format(config.acer_lr))
    print("ACER c_bar: {}".format(config.acer_c_bar))
    print("ACER replay size: {}".format(config.acer_replay_size))
    print("ACER replay ratio: {}".format(config.acer_replay_ratio))
    print("ACER trust-region delta: {}".format(config.acer_trust_region_delta))
    print("ACER EMA alpha: {}".format(config.acer_ema_alpha))
    print("ACER value coef: {}".format(config.acer_value_coef))
    print("ACER entropy coef: {}".format(config.acer_entropy_coef))
    print("Device: {}".format(config.device_name))


def train_acer(sequences, scoring="sp", save=None, load=None, checkpoint=0,
               fasta_path=None, results_csv=None, figures_dir=None, no_compare=False,
               eval_interval=100, patience=0, min_delta=0.0, target_rank=None,
               entropy_end=None, inference_rollouts=1,
               pretrain=False, pretrain_epochs=100):
    """Train an ACER agent on the provided sequences.

    Loop structure:
        for each episode:
            collect full episode → agent.select() + agent.record_transition()
            on-policy update    → agent.update()
                (also performs acer_replay_ratio off-policy updates from replay
                 buffer and updates the EMA average policy)
        run greedy inference    → agent.predict()

    ACER improves on A2C by:
      • Using Retrace(λ) Q-value estimates instead of Monte-Carlo returns,
        reducing variance while correcting for off-policy bias.
      • Replaying past episodes for better sample efficiency.
      • Adding a trust-region penalty (KL from EMA average policy) to prevent
        catastrophic policy updates.

    Weight persistence:
      • load  — resume from a previously saved checkpoint.
      • save  — write final weights after training completes.
      • checkpoint > 0 — additionally write weights/NAME_epN.pth every N episodes
                          so training can survive interruptions.
    """
    output_parameters_acer()
    train_start_time = time.monotonic()

    print(f"Training on {len(sequences)} sequences (ACER):")
    for key in sequences:
        print(f"Sequence {key}")

    # Entropy annealing: if entropy_end is set, decay exponentially from the
    # current config.acer_entropy_coef down to entropy_end over the full run.
    entropy_start = config.acer_entropy_coef
    _do_anneal    = (entropy_end is not None and entropy_end < entropy_start
                     and config.max_episode > 1)
    if _do_anneal:
        print(f"  Entropy annealing: {entropy_start:.4f} → {entropy_end:.4f} "
              f"over {config.max_episode} episodes")

    seqs_list = list(sequences.values())
    env       = Environment(seqs_list)
    agent     = ACER(env.action_number, env.row, env.max_len)
    _maybe_load(agent, load)

    # ── Imitation-learning pretraining (optional) ─────────────────────────────
    if pretrain:
        if not fasta_path:
            print("  [Pretrain] --pretrain requires --fasta-path; skipping")
        else:
            print(f"  [Pretrain] Running MAFFT to obtain expert alignment ...")
            pt_score, pt_aligned = _run_mafft(fasta_path, seqs_list,
                                              return_alignment=True)
            if pt_aligned is None:
                print("  [Pretrain] MAFFT unavailable — skipping pretraining")
            else:
                expert_data = _decode_alignment_to_actions(env, pt_aligned)
                _pretrain_acer(agent, expert_data, pretrain_epochs)

    classical = _bench_classical_scores(fasta_path) if fasta_path else []
    p         = tqdm(range(config.max_episode))

    best_score, patience_counter = -float("inf"), 0
    ep = 0
    for ep in p:
        # ── Entropy annealing ─────────────────────────────────────────────────
        if _do_anneal:
            frac = ep / max(1, config.max_episode - 1)
            config.acer_entropy_coef = entropy_start * (
                (entropy_end / entropy_start) ** frac)

        state = env.reset()
        while True:
            action = agent.select(state)
            reward, next_state, done = env.step(action)
            agent.record_transition(reward, float(done))
            if done == 0:
                break
            state = next_state
        agent.update()
        _maybe_checkpoint(agent, save, checkpoint, ep)

        if eval_interval > 0 and (ep + 1) % eval_interval == 0:
            _, best_score, patience_counter, stop = _check_early_stop(
                agent, seqs_list, ep, best_score, patience_counter,
                patience, min_delta, classical, target_rank, save, p)
            if stop:
                break

    # Restore entropy to final annealed value (or original) for reporting
    if _do_anneal:
        config.acer_entropy_coef = entropy_end

    episodes_run = ep + 1
    train_end_time = time.monotonic()
    _maybe_save(agent, save)
    print(f"Training Complete (ACER) — {episodes_run} episodes")
    print(f"Training time: {train_end_time - train_start_time:.2f} seconds")

    # ── Restore best checkpoint before inference ──────────────────────────────
    # _check_early_stop saves {save}_best whenever a new peak SP is found.
    # Load it now so inference runs on the best weights, not the final weights.
    if save:
        best_ckpt = f"{save}_best"
        best_ckpt_path = os.path.join(_weight_dir(), best_ckpt + ".pth")
        if os.path.exists(best_ckpt_path):
            agent.load(best_ckpt, path=_weight_dir())
            print(f"[INFO] Loaded best checkpoint '{best_ckpt}' for inference.")

    # ── Inference: best-of-N stochastic rollouts ──────────────────────────────
    # With inference_rollouts=1 (default) this is equivalent to the old single
    # greedy rollout via agent.predict().  With N≥2 we sample N full episodes
    # from the trained stochastic policy and keep the highest-scoring alignment.
    predict_start_time = time.monotonic()
    n_rollouts = max(1, inference_rollouts)
    best_inf_score  = -float("inf")
    best_inf_aligned = None

    for _ in range(n_rollouts):
        state = env.reset()
        if n_rollouts == 1:
            # Deterministic greedy rollout (original behaviour)
            while True:
                action = agent.predict(state)
                _, next_state, done = env.step(action)
                state = next_state
                if done == 0:
                    break
        else:
            # Stochastic rollout — samples from π_θ
            while True:
                action = agent.select(state)
                _, next_state, done = env.step(action)
                agent.record_transition(0.0, float(done))  # dummy reward
                state = next_state
                if done == 0:
                    break

        env.padding()
        rollout_score = env.calc_score()
        if rollout_score > best_inf_score:
            best_inf_score  = rollout_score
            best_inf_aligned = [list(seq) for seq in env.aligned]

    # Restore the best alignment into env for _print_results / calc_score
    if best_inf_aligned is not None:
        env.aligned = best_inf_aligned

    predict_end_time = time.monotonic()
    if n_rollouts > 1:
        print(f"Prediction Complete (ACER) — best of {n_rollouts} rollouts, "
              f"SP = {best_inf_score}")
    else:
        print("Prediction Complete (ACER)")
    print(f"Predict time: {predict_end_time - predict_start_time:.2f} seconds")

    # env.aligned holds the best rollout; padding was applied inside the loop
    sp_score = env.calc_score()
    _print_results(env, scoring)
    print("********************************\n")
    if fasta_path:
        mafft_score = _run_mafft(fasta_path, seqs_list)
        _run_benchmark_comparison(fasta_path, sp_score,
                                  episodes_run,
                                  train_end_time - train_start_time,
                                  results_csv, figures_dir, no_compare,
                                  mafft_score=mafft_score)


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
