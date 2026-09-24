import multiprocessing as mp
import queue


# Hard-code your graph labels here
LABELS = [
    "Reward",
    "Error Rate",
    "Loss",
    "cheese"
]


class StatsDashboard:
    def __init__(self):
        self.connection = None
        self.process = None

    def start(self):
        self.connection = mp.Queue()

        self.process = mp.Process(
            target=setup_graph,
            args=(self.connection,)
        )

        self.process.start()

    def send(self, stats):
        if len(stats) != len(LABELS):
            raise ValueError(
                f"Expected {len(LABELS)} stats, got {len(stats)}"
            )

        self.connection.put(stats)

    def stop(self):
        self.connection.put(None)
        self.process.join()


def setup_graph(connection):
    import matplotlib.pyplot as plt

    histories = [
        [] for _ in LABELS
    ]

    graph = 0

    plt.ion()

    fig, ax = plt.subplots()
    line, = ax.plot([], [])

    def change_graph(event):
        nonlocal graph

        if event.key == "right":
            graph = (graph + 1) % len(LABELS)

        elif event.key == "left":
            graph = (graph - 1) % len(LABELS)

    fig.canvas.mpl_connect(
        "key_press_event",
        change_graph
    )

    running = True

    while running and plt.fignum_exists(fig.number):

        try:
            while True:
                stats = connection.get_nowait()

                if stats is None:
                    running = False
                    break

                for i in range(len(stats)):
                    histories[i].append(stats[i])

        except queue.Empty:
            pass

        values = histories[graph]

        line.set_data(
            range(1, len(values) + 1),
            values
        )

        ax.set_title(
            f"{LABELS[graph]}   ({graph + 1}/{len(LABELS)})"
        )

        ax.set_xlabel("Episode")
        ax.set_ylabel(LABELS[graph])

        ax.relim()
        ax.autoscale_view()

        fig.canvas.draw_idle()
        plt.pause(0.05)

    plt.close()

if __name__ == "__main__":

    graphs = StatsDashboard()
    graphs.start()

    graphs.send([
        20.5,
        0.23,
        2.7,
        3     
    ])

    graphs.send([
        28.2,
        0.17,
        2.1,
        3
    ])

    graphs.send([
        40.8,
        0.09,
        1.3,
        2
    ])

    graphs.process.join()