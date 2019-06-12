#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
#from functools import reduce


# #### Read Data

# In[2]:


path = '/home/seherkhan/myfiles/coursework/usc/spring2019/inf553/proj/io/'
# Projects.csv
projects_df = pd.read_csv(path+"Projects.csv")
projects_df.drop_duplicates(['Project ID'],inplace=True)
#projects_df.head()

# Donations.csv
donations_df = pd.read_csv(path+"Donations.csv")
donations_df.drop_duplicates(['Donation ID'],inplace=True)
# comment the following if you do not want to sum donations of same donor to same project
donations_df = donations_df[['Project ID','Donor ID','Donation Amount']].groupby(['Project ID','Donor ID'], as_index=False).agg({'Donation Amount':np.sum})
#donations_df.head()

# Donors.csv
donors_df=pd.read_csv(path+"Donors.csv")
donors_df.drop_duplicates(inplace=True)
#donors_df.head()


# #### Join Tables

# In[4]:


#Merge donations_df, donors_df, projects_df
donors_df_cols=['Donor ID','Donor City']
donations_df_cols=['Donor ID','Project ID','Donation Amount'] # 'Donation ID'
projects_df_cols=['Project ID','Project Title','Project Type','Project Resource Category','Project Subject Category Tree','Project Subject Subcategory Tree']

all1=pd.merge(donations_df[donations_df_cols],donors_df[donors_df_cols],
                 on='Donor ID', 
                 how='outer')
all2=pd.merge(all1,projects_df[projects_df_cols],
                 on='Project ID', 
                 how='outer')
# all2.keys() # columns in all2
# Index([u'Donor ID', u'Project ID', u'Donation Amount', u'Donor City',
#       u'Project Type', u'Project Resource Category',
#       u'Project Subject Category Tree', u'Project Subject Subcategory Tree',
#       u'Project Resource Category'],
#      dtype='object')


# In[8]:


#'Project Type' # ['Teacher-Led', 'Professional Development', 'Student-Led']
#'Project Subject Category Tree' # 52 categories e.g. ['Applied Learning', 'Applied Learning, Literacy & Language']
#'Project Subject Subcategory Tree' # 433 categories e.g. ['Character Education, Early Development','Early Development, Literacy']
#'Project Resource Category' # 18 categories e.g. ['Technology', 'Supplies', 'Books']

#len(projects_df[['Project Type','Project Resource Category','Project Subject Category Tree','Project Subject Subcategory Tree','Project Resource Category']].drop_duplicates())
#len(projects_df['Project ID'].drop_duplicates()) 
# 7551 distinct "types" of projects
# 1110015 distinct projects

# ensured that none of these categories contain the pipeline character
# all2['Project Subject Subcategory Tree'].str.contains(r'\|').sum()


# #### Take subset for desired city

# In[5]:


# If want to work with all donors, uncomment the following line
#city_df = all2

# City of Donors to focus recommendations on 
city="Oakland"
city_df = all2[all2['Donor City']==city]
print city_df.keys()
print city_df.shape
print len(city_df['Donor ID'].unique())
print len(city_df['Project ID'].unique())

df_presentation = all2[['Donor City','Donor ID', 'Project ID', 'Donation Amount','Project Title','Project Type','Project Resource Category',
       'Project Subject Category Tree', 'Project Subject Subcategory Tree']].head()
df_presentation.head()
df_presentation[df_presentation['Donor ID']=='a5c69797ed95ffa7f18bc69e8540c676']all2[all2['Donor ID']=='a5c69797ed95ffa7f18bc69e8540c676']
# #### Save helper datasets which will be useful for LSH

# In[ ]:


# following data set will be used to find recommendations in LSH
f3 =open('Oakland_dataset.csv','w')
city_df.to_csv(f3)
f3.close()


# In[6]:


# following dataset will be used to make LSH model
ProjTypeID_Definition = city_df[['Project ID','Project Type','Project Resource Category',
       'Project Subject Category Tree','Project Subject Subcategory Tree']].drop_duplicates(subset=['Project Type','Project Resource Category',
       'Project Subject Category Tree','Project Subject Subcategory Tree'],keep='last')
ProjTypeID_Definition['projtype_id'] = np.arange(len(ProjTypeID_Definition))
print ProjTypeID_Definition.shape
print ProjTypeID_Definition.keys()
ProjTypeID_Definition.head()
f1=open('proj_def_oakland_1.csv','w')
ProjTypeID_Definition.iloc[:,1:].to_csv(f1)
f1.close()

# assumed each project belongs to only one category
# since RDBS, assumed projects ordered least recent to most recent 
# (appears to be so from "Project Posted Date" column)


# #### Construction of Binary Utility Matrix

# In[7]:


city_df_projtypeid = city_df[['Project ID','Donor ID']].merge(ProjTypeID_Definition[['Project ID','projtype_id']], 
                                   on=['Project ID'], how="left") \
                            .drop(['Project ID'],axis=1).drop_duplicates().dropna().astype({'projtype_id':int})
# Note: dropped donors that have not donated to any project since 
# they cannot be recommended projects with collaborative filtering

city_df_projtypeid['pivot_dummy']=1

print city_df_projtypeid.shape
city_df_projtypeid.head(50)len(city_df_projtypeid['Donor ID'].unique())
# In[9]:


utility_mat = city_df_projtypeid.pivot_table(index='Donor ID', columns='projtype_id',values ='pivot_dummy').fillna(0)
utility_mat.head()


# In[10]:


# Write utility matrix to csv
f=open('ulit_mat_oakland_projtypeid_binary_1.csv','w')
utility_mat.to_csv(f)
f.close()


# #### Construction of Ratings Matrix

# In[28]:


city_df_projtypeid_forratings = city_df[['Project ID','Donor ID','Donation Amount']].merge(ProjTypeID_Definition[['Project ID','projtype_id']], 
                                   on=['Project ID'], how="left") \
                            .drop(['Project ID'],axis=1).drop_duplicates().dropna().astype({'projtype_id':int})
# Note: dropped donors that have not donated to any project since 
# they cannot be recommended projects with collaborative filtering


# In[29]:


# Create columns project_id, donor_id, Rating by Donor, Rating by All Donors
tmp_donor = city_df_projtypeid_forratings.groupby(['Donor ID'], as_index=False).agg({'Donation Amount':np.sum}).rename(columns={'Donation Amount': 'Donations by Donor'})

#tmp_proj = city_df_projtypeid_forratings.groupby(['projtype_id'], as_index=False).agg({'Donation Amount':np.sum}).rename(columns={'Donation Amount': 'Donations to ProjectType'})

city_df_projtypeid_forratings = city_df_projtypeid_forratings.merge(tmp_donor, on=['Donor ID'], how="left")
tmp_donor=None
#city_df_projtypeid_forratings = city_df_projtypeid_forratings.merge(tmp_proj, on=['projtype_id'], how="left")
#tmp_proj=None

city_df_projtypeid_forratings['Rating by Donor']=city_df_projtypeid_forratings['Donation Amount']/city_df_projtypeid_forratings['Donations by Donor']
#city_df_projtypeid_forratings['Rating by All Donors']=city_df_projtypeid_forratings['Donation Amount']/city_df_projtypeid_forratings['Donations to ProjectType']

city_df_projtypeid_forratings.head()


# In[38]:


ratingmatrix_this_donor = city_df_projtypeid_forratings.pivot_table(index='Donor ID', columns='projtype_id',values = 'Rating by Donor').fillna(0)


# In[39]:


ratingmatrix_this_donor.head()


# In[40]:


# Write ratings matrix to csv
f4=open('rating_mat_oakland_projtypeid_1.csv','w')
ratingmatrix_this_donor.to_csv(f4)
f4.close()

