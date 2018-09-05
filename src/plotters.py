import numpy as np
import pandas as pd
import matplotlib
#matplotlib.use('agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from nltk.stem.porter import PorterStemmer
from datetime import timedelta


def time_food_plot(df, food, n_months, path=None, save=False):
    porter = PorterStemmer()
    food_stem = porter.stem(food.lower())

    dates = df.loc[[i for i in df.index if food_stem in df.loc[i, 'food_stems']], ['id', 'post_date']]
    dates.index = dates['post_date']
    dates['food_counts'] = np.ones(len(dates))
    counts = dates.resample('{}M'.format(n_months)).sum()
    counts['food_counts'].fillna(0, inplace=True)

    dates_all = df[['id', 'post_date']]
    dates_all.index = dates_all['post_date']
    dates_all['total_counts'] = np.ones(len(dates_all))
    counts_all = dates_all.resample('1M').sum()
    counts_all['total_counts'].fillna(0, inplace=True)

    final = counts[['food_counts']].join(counts_all[['total_counts']], how='left')
    final.fillna(0, inplace=True)
    final['percentage'] = final['food_counts'] / final['total_counts']

    yearly_counts = final.resample('1Y').sum()
    yearly_counts['yearly_percentage'] = yearly_counts['food_counts']/yearly_counts['total_counts']
    yearly_counts.index = yearly_counts.index - timedelta(days = 180)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(final.index, final['percentage'], color='purple', alpha=0.5, linewidth=2)
    ax.plot(yearly_counts.index, yearly_counts['yearly_percentage'], color='blue', linewidth=4)
    ax.set_title('Popularity of {} Over Time'.format(food))
    ax.set_xlabel('Year')
    if save:
        fig.savefig(path)
    return fig


def yearly_food_plot(df, food, all_plots=True, path=None, save=False):
    porter = PorterStemmer()
    food_stem = porter.stem(food.lower())
    dates = df.loc[[i for i in df.index if food_stem in df.loc[i, 'food_stems']], ['id', 'post_date']]
    dates.index = dates['post_date']
    dates['food_counts'] = np.ones(len(dates))

    dates_all = df[['id', 'post_date']]
    dates_all.index = dates_all['post_date']
    dates_all['total_counts'] = np.ones(len(dates_all))

    months = np.array(range(1, 13))
    mo_counts = np.zeros((len(range(2009, 2018)), len(months)))
    mo_counts_all = np.zeros((len(range(2009, 2018)), len(months)))

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, year in enumerate(range(2009, 2018)):
        year = str(year)
        d1 = dates[year].resample('1M').sum()
        d2 = dates_all[year].resample('1M').sum()
        counts = d1[['food_counts']].join(d2[['total_counts']], how='left')
        counts.fillna(0, inplace=True)
        counts['percentage'] = counts['food_counts'] / counts['total_counts']
        counts['mos'] = [date.month for date in counts['food_counts'].index]
        mo_counts[i][counts['mos'] -1] = counts['percentage']
        if all_plots:
            ax.plot(months, mo_counts[i],
                linewidth=2,
                color=plt.cm.cool(10 + i*30),
                alpha=0.3,
                label=year)

    avg_counts = np.mean(mo_counts, axis=0)
    ax.plot(months, avg_counts, linewidth=4, color='blue', label='2009-2017\nAverage')
    ax.set_xlim([0, 15])
    ax.set_xticks([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.set_title('Seasonal Popularity of {}'.format(food))
    ax.set_xlabel('Month')
    ax.legend()
    if save:
        fig.savefig(path)
    return fig
