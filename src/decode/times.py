"""
    Class dedicated to testing some processing times for my approach within the tree architecture
"""

import time
from collections import defaultdict

class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self._start_times = {}

    def start(self, label: str):
        """
            Start timing a labeled task 
            Args:
                - label (str): name of the task being monitored
        """
        self._start_times[label] = time.perf_counter()

    def stop(self, label):
        """
            Stop timing and accumulate elapsed time for a labeled task
            Args:
                - label (str): name of the task being monitored
        """
        if label not in self._start_times:
            raise ValueError(f"Timer for '{label}' was not started.")
        elapsed = time.perf_counter() - self._start_times.pop(label)
        self.times[label] += elapsed

    def report(self):
        """
            Return a dict with all accumulated times
        """
        
        total = sum(self.times.values())
        report = {label: f"{t:.4f}s ({t/total*100:.1f}%)" for label, t in self.times.items()}
        report["TOTAL"] = f"{total:.4f}s"
        return report