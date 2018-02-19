import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nltk.stem.porter import PorterStemmer


def time_food_plot(df, food, n_months, path=None, save=True):
    porter = PorterStemmer()
    food_stem = porter.stem(food.lower())
    dates = df.loc[[i for i in df.index if food_stem in df.loc[i, 'food_stems']], ['id', 'post_date']]
    dates.index = dates['post_date']
    dates['ones'] = np.ones(len(dates))
    counts = dates.resample('{}M'.format(n_months)).sum()
    counts['ones'].fillna(0, inplace=True)
    fig, ax = plt.subplots(figsize=(20, 5))
    counts['ones'].plot(ax=ax)
    ax.set_title('Popularity of {}'.format(food))
    if save:
        fig.savefig(path)


def yearly_food_plot(df, food, path=None, save=True):
    porter = PorterStemmer()
    food_stem = porter.stem(food.lower())
    dates = df.loc[[i for i in df.index if food_stem in df.loc[i, 'food_stems']], ['id', 'post_date']]
    dates.index = dates['post_date']
    dates['ones'] = np.ones(len(dates))
    months = np.array(range(1, 13))
    mo_counts = np.zeros((len(range(2009, 2018)), len(months)))
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, year in enumerate(range(2009, 2018)):
        year = str(year)
        counts2 = dates.resample('1M').sum()
        yearly_food = counts2[year].resample('1M').sum()
        yearly_food.fillna(0, inplace=True)
        yearly_food['mos'] = [date.month for date in yearly_food['ones'].index]
        mo_counts[i][yearly_food['mos'] -1] = yearly_food['ones']
        ax.plot(months, mo_counts[i],
                linewidth=2,
                color=plt.cm.cool(10 + i*30),
                alpha=0.3,
                label=year)
    avg_counts = np.mean(mo_counts, axis=0)
    ax.plot(months, avg_counts, linewidth=4, color='blue')
    ax.set_xlim([0, 15])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_title('Average Yearly Popularity of {}'.format(food))
    ax.set_xlabel('Month')
    ax.legend()
    if save:
        fig.savefig(path)
