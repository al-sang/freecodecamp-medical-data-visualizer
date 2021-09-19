from numpy.lib.twodim_base import triu_indices_from
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Import data
df = pd.read_csv('medical_examination.csv')

# Clean data
df = df[df['ap_lo'] <= df['ap_hi']]
df = df[df['height'] >= df['height'].quantile(0.025)]
df = df[df['height'] <= df['height'].quantile(0.975)]
df = df[df['weight'] >= df['weight'].quantile(0.025)]
df = df[df['weight'] <= df['weight'].quantile(0.975)]
df.reset_index(drop=True, inplace=True)

# Add 'overweight' column
df['bmi'] = round(df['weight']/((df['height']/100)**2), 3)
df['overweight'] = np.where((df['bmi'] > 25), 1, 0)

# Normalize data by making 0 always good and 1 always bad. If the value of 'cholesterol' or 'gluc' is 1, make the value 0. If the value is more than 1, make the value 1.
df['cholesterol'] = np.where((df['cholesterol'] == 1), 0, 1)
df['gluc'] = np.where((df['gluc'] == 1), 0, 1)

# Draw Categorical Plot


def draw_cat_plot():
    # Create DataFrame for cat plot using `pd.melt` using just the values from 'cholesterol', 'gluc', 'smoke', 'alco', 'active', and 'overweight'.
    df_cat = df.melt(id_vars='cardio', value_vars=df[[
                     'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight']])

    # Group and reformat the data to split it by 'cardio'. Show the counts of each feature. You will have to rename one of the columns for the catplot to work correctly.
    #df_cat = None

    # Draw the catplot with 'sns.catplot()'
    fig = sns.catplot(data=df_cat, x='variable',
                      kind='count', col='cardio', hue='value')
    # Do not modify the next two lines
    fig.savefig('catplot.png')
    return fig


# Draw Heat Map
def draw_heat_map():
    # Clean the data
    df_heat = df

    # Calculate the correlation matrix
    corr = df.corr().round(decimals=1)

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    fig, ax = plt.subplots()

    # Draw the heatmap with 'sns.heatmap()'
    sns.heatmap(corr, annot=True, center=0, linewidths=0.5, mask=mask)
    # Do not modify the next two lines
    fig.savefig('heatmap.png', facecolor=fig.get_facecolor())
    return fig
