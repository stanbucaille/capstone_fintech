import collections
import pandas as pd
import numpy as np


class ObservationHistory(object):

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.rows = pd.DataFrame()

    def push(self, row: dict):
        """Saves an observation."""
        self.rows = self.rows.append(row, ignore_index=True)

        if len(self.rows) > self.window_size:
            self.rows = self.rows[-self.window_size:]

    def observe(self) -> np.array:
        """Returns the rows to be observed by the agent."""
        rows = self.rows.copy()
        
        #if not enought rows, padding by 0 entries
        if len(rows) < self.window_size:
            size = self.window_size - len(rows)
            padding = np.zeros((size, rows.shape[1]))
            padding = pd.DataFrame(padding, columns=self.rows.columns)
            rows = pd.concat([padding, rows], ignore_index=True, sort=False)

        rows = rows.fillna(0).values   #not a good preprocessing scheme

        return rows

    def reset(self):
        self.rows = pd.DataFrame()
