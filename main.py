# FOR THE FINAL TRAINING LOOP

from agent_ppo import StartAgent

if __name__ == "__main__":
    final_network = StartAgent(48, 24, 24, 9000, 29, "orth", 0.5, 0, 360, None)