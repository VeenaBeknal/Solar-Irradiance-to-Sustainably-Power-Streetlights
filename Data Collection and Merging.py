#!/usr/bin/env python
# coding: utf-8

# In[40]:


import os
import pandas as pd
import glob
import matplotlib.pyplot as plt


# In[17]:


# Directory containing folders with zip code names
base_dir = r'D:\Veena\SJSU-Classes\Sem2\DATA-245 Sec 21 - Machine Learning\Project\Data\Sanjose Data'


# In[18]:


# Initializing an empty list to store dataframes
dfs = []

# Iterating over each folder in the base directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Checking if the path is a directory
    if os.path.isdir(folder_path):
        # Extract zip code from folder name
        zipcode = folder_name
        
        # Getting a list of CSV files in the current folder
        csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
        
        # Iterating over each CSV file
        for file in csv_files:
            # Reading CSV file into a dataframe skipping first 2 rows
            df = pd.read_csv(file, skiprows=2)
            
            # Adding a column to capture the zipcode
            df['zipcode'] = zipcode
            
            # Appending the dataframe to the list
            dfs.append(df)

# Concatenating all dataframes into a single dataframe
final_df = pd.concat(dfs, ignore_index=True)


# In[19]:


# Printing the final dataframe
print(final_df)


# In[20]:


# Checking to see if it has worked as expected
final_df.tail


# In[21]:


# Creating hour_decimal variable in decimal format for ease of day part caculation
final_df['hour_decimal'] = final_df['Hour'] + final_df['Minute'] / 60


# In[22]:


print(final_df.head())


# In[23]:


# Defining function to map hour values to day parts
def map_hour_to_day_part(hour_decimal):
    if hour_decimal >= 0 and hour_decimal < 5:
        return 'late_night_start_of_day'
    elif hour_decimal >= 5 and hour_decimal < 8:
        return 'early_morning'
    elif hour_decimal >= 8 and hour_decimal < 12:
        return 'morning'
    elif hour_decimal >= 12 and hour_decimal < 16:
        return 'afternoon'
    elif hour_decimal >= 16 and hour_decimal < 18:
        return 'evening'
    elif hour_decimal >= 18 and hour_decimal < 20:
        return 'late_evening'
    elif hour_decimal >= 20 and hour_decimal < 22:
        return 'early_night'
    else:
        return 'late_night'


# In[24]:


# Applying the function to map hour to day part and create hour_day_part column
final_df['hour_day_part'] = final_df['hour_decimal'].apply(map_hour_to_day_part)


# In[25]:


# Dropping the intermediate 'hour_decimal' column since it's not required
final_df.drop(columns=['hour_decimal'], inplace=True)

# Checking the new column
print(final_df)


# In[26]:


final_df_csv = pd.DataFrame(final_df)

# Save DataFrame to a CSV file
final_df_csv.to_csv('SolarIrradiance.csv', index=False)  # Set index=False to exclude index column in the CSV file

print("DataFrame converted to CSV successfully.")


# In[27]:


final_df['zipcode'].unique()


# In[29]:


zipcode_counts = final_df.groupby('zipcode').size()
zipcode_counts
# 365 days * 24 hours * 5 years = 43800 + 24 extra rows for leap year (2020)
# the correct value is 43824


# In[32]:


# Creating a working copy for reference
working_ir_df = final_df.copy()


# In[33]:


working_ir_df.dtypes


# In[34]:


# Defining columns to be dropped
columns_to_drop = ['Fill Flag', 'Surface Albedo', 'Wind Direction', 'Global Horizontal UV Irradiance (280-400nm)', 
                  'Global Horizontal UV Irradiance (295-385nm)']

# Dropping multiple unnecessary columns
working_ir_df.drop(columns=columns_to_drop, inplace=True)


# In[35]:


working_ir_df.describe()


# In[36]:


# Save DataFrame to a CSV file
working_ir_df.to_csv('SolarIrradiance - processed.csv', index=False)  

print("DataFrame converted to CSV successfully.")

