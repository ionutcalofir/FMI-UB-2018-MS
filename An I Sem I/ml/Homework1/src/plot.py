import numpy as np
import matplotlib.pyplot as plt

from dataset import Dataset

class Plot():
    def __init__(self):
        self._data_df = Dataset().get_dataset
        self._X = self._data_df.values
        self._fields = self._data_df.keys().values
        self._fields_dict = {k: v for (v, k) in enumerate(self._data_df.keys())}

    def plot_missing_values(self):
        values = []
        ticks = []
        for idx, field in enumerate(self._fields):
            pos_field = self._fields_dict[field]
            value = 0
            try:
                if np.isnan(self._X[:, pos_field]).any():
                    names, counts = np.unique(class_data[:, pos_field], return_counts=True)
                    for name, count in zip(names, counts):
                        if name == 'nan':
                            value = count
            except: # column is categorical
                for val in self._X[:, pos_field]:
                    if not isinstance(val, str) and np.isnan(val):
                        value += 1

            values.append(value)
            ticks.append(str(idx) + ' - ' + field)

        fig = plt.figure('Missing Values')
        fig.suptitle('Missing Values')
        ax = fig.subplots()
        ax_bars = ax.bar(np.arange(len(values)), values)
        ax.set_xticks(np.arange(len(values)))
        ax.set_xlabel('Fields')
        ax.set_ylabel('Counts')
        ax.legend(ax_bars, ticks, loc=2, fontsize='xx-small')

        plt.show()
