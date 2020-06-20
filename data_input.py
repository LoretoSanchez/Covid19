"""
Data massaging: prepare the data so that it is easy to plot it.
"""

import os
import pickle
import pandas as pd


def get_populations():
    """ Load the information that we have about countries """
    pop = pd.read_csv('data/countryInfo.txt', sep='\t', skiprows=50)
    return pop


def normalize_by_population(tidy_df):
    """ Normalize by population the column "Contagios" of a dataframe with
        lines being the country ISO
    """
    pop = get_populations()

    pop0 = pop.set_index('ISO3')['Population']
    contagios = df_recent.set_index('iso')['Contagios']
    divisor=contagios.copy()
    for idx,i in enumerate(df_recent['iso']):
        divisor[idx] = pop0[i]

    normalized_values = (df_recent.set_index('iso')['Contagios']
                         / divisor)

    # NAs appeared because we don't have data for all entries of the pop
    # table
    normalized_values = normalized_values.dropna()
    assert len(normalized_values) == len(tidy_df),\
        ("Not every country in the given dataframe was found in our "
         "database of populations")
    return normalized_values



