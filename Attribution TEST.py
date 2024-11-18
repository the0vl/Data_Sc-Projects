#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns
from collections import defaultdict
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


import os
os.getcwd()


# In[4]:


df = pd.read_csv('attribution data.csv')


# In[5]:


df.head()


# In[6]:


df['interaction'].unique()


# In[7]:


df = df.sort_values(['cookie', 'time'],
                    ascending=[False, True])
df['visit_order'] = df.groupby('cookie').cumcount() + 1


# In[8]:


df[df['visit_order'] == df['visit_order'].max()]


# In[15]:


df.head(100)


# In[9]:


df_paths = df.groupby(['cookie'], as_index= False)['channel'].aggregate(lambda x: x.unique().tolist())


# In[13]:


df_paths.head()


# In[10]:


df_last_interaction = df.drop_duplicates('cookie', keep='last')[['cookie', 'conversion']]
df_paths = pd.merge(df_paths, df_last_interaction, how='left', on='cookie')


# In[11]:


df_paths['conversion'].value_counts()


# In[12]:


df_paths


# In[14]:


df_paths['path'] = np.where(
 df_paths['conversion'] == 0,
 ['Start, '] + df_paths['channel'].apply(', '.join) + [', Null'],
 ['Start, '] + df_paths['channel'].apply(', '.join) + [', Conversion'])
df_paths['path'] = df_paths['path'].str.split(', ')


# In[15]:


df_paths = df_paths[['cookie', 'path']]


# In[16]:


df_paths


# In[17]:


list_of_paths = df_paths['path']
total_conversions = sum(path.count('Conversion') for path in df_paths['path'].tolist())
base_conversion_rate = total_conversions / len(list_of_paths)


# In[18]:


# print(list_of_paths)
print(f'total conversions:\n{total_conversions}')
print(f'base conversion rate:\n{base_conversion_rate}')


# In[19]:


def transition_states(list_of_paths):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    transition_states = {x + '>' + y: 0 for x in list_of_unique_channels for y in list_of_unique_channels}

    for possible_state in list_of_unique_channels:
        if possible_state not in ['Conversion', 'Null']:
            for user_path in list_of_paths:
                if possible_state in user_path:
                    indices = [i for i, s in enumerate(user_path) if possible_state in s]
                    for col in indices:
                        transition_states[user_path[col] + '>' + user_path[col + 1]] += 1

    return transition_states


trans_states = transition_states(list_of_paths)


# In[20]:


trans_states


# In[21]:


def transition_prob(trans_dict):
    list_of_unique_channels = set(x for element in list_of_paths for x in element)
    trans_prob = defaultdict(dict)
    for state in list_of_unique_channels:
        if state not in ['Conversion', 'Null']:
            counter = 0
            index = [i for i, s in enumerate(trans_dict) if state + '>' in s]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    counter += trans_dict[list(trans_dict)[col]]
            for col in index:
                if trans_dict[list(trans_dict)[col]] > 0:
                    state_prob = float((trans_dict[list(trans_dict)[col]])) / float(counter)
                    trans_prob[list(trans_dict)[col]] = state_prob

    return trans_prob


trans_prob = transition_prob(trans_states)


# In[22]:


len(trans_prob)
trans_prob


# In[23]:


def transition_matrix(list_of_paths, transition_probabilities):
    trans_matrix = pd.DataFrame()
    list_of_unique_channels = set(x for element in list_of_paths for x in element)

    for channel in list_of_unique_channels:
        trans_matrix[channel] = 0.00
        trans_matrix.loc[channel] = 0.00
        trans_matrix.loc[channel][channel] = 1.0 if channel in ['Conversion', 'Null'] else 0.0

    for key, value in transition_probabilities.items():
        origin, destination = key.split('>')
        trans_matrix.at[origin, destination] = value

    return trans_matrix


trans_matrix = transition_matrix(list_of_paths, trans_prob)


# In[24]:


trans_matrix


# In[39]:


sns.heatmap(trans_matrix.drop(['Null', 'Start','Conversion']).drop(['Null', 'Start', 'Conversion'], axis = 1), cmap = 'RdYlGn')


# In[41]:


from ast import Global


def removal_effects(df, conversion_rate):
    removal_effects_dict = {}
    channels = [channel for channel in df.columns if channel not in [
                'Start',
                'Null',
                'Conversion']]
    for channel in channels:
        removal_df = df.drop(channel, axis=1).drop(channel, axis=0)
        for column in removal_df.columns:
            row_sum = np.sum(list(removal_df.loc[column]))
            null_pct = float(1) - row_sum
            if null_pct != 0:
                removal_df.loc[column]['Null'] = null_pct
            removal_df.loc['Null']['Null'] = 1.0

        removal_to_conv = removal_df[
            ['Null', 'Conversion']].drop(['Null', 'Conversion'], axis=0)
        removal_to_non_conv = removal_df.drop(
            ['Null', 'Conversion'], axis=1).drop(['Null', 'Conversion'], axis=0)

        removal_inv_diff = np.linalg.inv(
            np.identity(
                len(removal_to_non_conv.columns)) - np.asarray(removal_to_non_conv))
        removal_dot_prod = np.dot(removal_inv_diff, np.asarray(removal_to_conv))
        removal_cvr = pd.DataFrame(removal_dot_prod,
                                   index=removal_to_conv.index)[[1]].loc['Start'].values[0]
        removal_effect = 1 - removal_cvr / conversion_rate
        removal_effects_dict[channel] = removal_effect

    return removal_effects_dict
    


removal_effects_dict = removal_effects(trans_matrix, base_conversion_rate)


# In[42]:


Markov_probabilities = pd.DataFrame(removal_effects_dict, index = [0])


# In[43]:


Markov_probabilities


# In[44]:


plt.figure(figsize= (14,8))
sns.barplot(x = Markov_probabilities.columns, y = Markov_probabilities.sum())
plt.show()


# shapley value

# In[1]:


import os


# In[6]:


os.getcwd()


# In[ ]:




