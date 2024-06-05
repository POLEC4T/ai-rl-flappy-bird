import argparse
from agent import train_dqn_agent, evaluate_dqn_agent

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DQN Agent for Flappy Bird")
    parser.add_argument('--mode', type=str, choices=['train', 'eval'], required=True, help="Mode to run: 'train' or 'eval'")
    parser.add_argument('--render', action='store_true', help="Render the game during evaluation")
    args = parser.parse_args()

    if args.mode == 'train':
        train_dqn_agent(render=args.render)
    elif args.mode == 'eval':
        evaluate_dqn_agent(render=args.render)
