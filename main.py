# FOR THE FINAL TRAINING LOOP

from agent_ppo import StartAgent
import time

start = time.perf_counter()
if __name__ == "__main__":
    final_network = StartAgent("test3000-256", 48, 24, 24, 9000, 29, "orth", 0.5, 2, 360, None)

print(f"finished: {time.perf_counter() - start}")