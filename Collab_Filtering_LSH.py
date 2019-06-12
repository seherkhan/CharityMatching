#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from pyspark import SparkContext
from itertools import combinations
from time import time as t
import operator


# In[2]:


'''Debugging
    - used to see if any chunk has only cells with zeros
    - calulates Sparsity of the matrix
'''
def num_non_zero_Mchunks(M_chunks):
    sum_all=0
    sum_nonzero=0
    print '#','all    ','nonzero'
    for i in range(len(M_chunks)):  
        print i,M_chunks[i].size,(M_chunks[i]!=0).sum()
        sum_all+=M_chunks[i].size
        sum_nonzero+=(M_chunks[i]!=0).sum()
    print 'T',sum_all,sum_nonzero
    print '---------------------'
    
    print '{0:.2f}'.format(100-(sum_nonzero*100/(1.0*sum_all)))+'% sparse'


# In[3]:


'''
    Reformats data in a particular chunk of M as a dictionary with 
    Donor IDs as keys and list of their ProjType IDs as values
'''
def get_projs_of_donors_in_M_chunk(chunk_id):
    M_chunk = M_chunks[chunk_id]
    M_chunk = M_chunks[chunk_id]
    n_users = M_chunk.shape[0] # 1303 donors
    n_projs = M_chunk.shape[1] # 179 projs
    users_projs={}
    for i in range(n_users): #n_users
        uid = str(chunk_id)+'_'+str(i)
        #print 'user',uid
        users_projs[uid]=[]
        for j in range(n_projs): #n_projs
            if M_chunk[i,j]==1:
                #print i,j
                users_projs[uid].append(j)
                #print 'add proj ',j
    return users_projs


# In[5]:


'''
    Hash function for computing signature
'''
def h(x,i):
    # i identifies the hash function
    # x is the row number (i.e. movie in question)
    # n_projs is the total number of movies
    return (5*x + 13*i) % n_projs

'''
    Function to compute minhash signature of donors in a chunk of M
'''
def compute_signature(chunk_id):
    M_chunk = M_chunks[chunk_id]
    n_users = M_chunk.shape[0]
    n_projs = M_chunk.shape[1]
    sig = [[inf for x in range(n_hashfunc)] for y in range(n_users)]
    # sig is a list of n_users_chunk = 975 lists, each of size n_hashfunc = 20
    for r in range(n_projs):
        h_for_r = [h(r,i) for i in range(0,n_hashfunc)] # i is hash function
        for c in range(n_users):
            #print M_chunk[c,r]
            if M_chunk[c,r] == 1:
                for j in range(n_hashfunc):
                    sig[c][j] = min(sig[c][j],h_for_r[j])
    sig_=[]
    for c in range(n_users):
        sig_.append((str(chunk_id)+'_'+str(c),sig[c])) # is the <chunk_id>_<donor_id>
    # sig_ is a list of items (i,signature_of_user_i_of_this_chunk)
    return sig_


# In[6]:


'''
    Helper method that calculates jaccard of two sets
    Returns zero when jaccard is zero or jaccard is undefined
'''
def jaccard(set1,set2):
    denom = float(len(set1.union(set2))) # union is empty when both donors have not donated to any projects
    if denom == 0.0:
        return 0.0
    else:
        return len(set1.intersection(set2))/float(len(set1.union(set2)))

'''
    Calculates jaccard of a donor pair
'''
def find_jaccard(pair):
    # get projects of the donors in the pair. do this as follows:
    #     take each donor in projs_of_donor
    #     if this donor appears in the pair, retain the donor
    #     next only retain the donors projects
    # finally, find jaccard of the two donors using their projects
    sets = map(lambda (donor_id,donor_projs): donor_projs,         filter(lambda (donor_id,donor_projs): donor_id==pair[0] or donor_id==pair[1],            projs_of_donors_in_chunks.items())     )
    return jaccard(set(sets[0]),set(sets[1]))


# In[69]:


'''
    Helper method to switch from actual Donor ID to lsh_donor_id used in this script
'''
def convert_donor_id_to_lsh(DONOR_ID):
    i_in_M = df.iloc[:,0][df.iloc[:,0]==DONOR_ID].index[0]
    uid=None
    for i in range(n_chunks):
        if i_in_M in M_ind_chunks[i]:
            return str(i)+'_'+str(np.where(M_ind_chunks[i]==i_in_M)[0][0])
    return None

'''
    Helper method to switch from lsh_donor_id used in this script to actual Donor ID
'''
def convert_donor_id_to_org(lsh_donor_id):
    MChunk_ind = int(lsh_donor_id.split('_')[0])
    if MChunk_ind ==0:
        M_ind = int(lsh_donor_id.split('_')[1])
    else:
        prev_chunks_count = 0
        for i in range(MChunk_ind):
            prev_chunks_count+=len(M_ind_chunks[i])
        M_ind = prev_chunks_count+int(lsh_donor_id.split('_')[1])
    #print M_ind
    return df.iloc[M_ind,0]

'''
    Helper Method to find similar donors given an lsh_donor_id
'''
def get_similar_donors(lsh_donor_id,returnLSHDonorIDs,rdd_similar_donors):
    similar_lsh_donors = rdd_similar_donors.lookup(lsh_donor_id)[0]
    #print similar_lsh_donors
    if returnLSHDonorIDs==True:
        return similar_lsh_donors
    else:
        return [convert_donor_id_to_org(lsh_donor_id) for lsh_donor_id in similar_lsh_donors]

'''
    Helper Method to find similar donors given an lsh_donor_id
'''
def get_projects_of_this_donor(DONOR_ID,returnDetails):
    if returnDetails:
        this_donor_proj_df = city_df[city_df['Donor ID']==DONOR_ID][['Project ID','Project Title','Project Type','Project Resource Category','Project Subject Category Tree','Project Subject Subcategory Tree']]
        this_donor_proj_df.reset_index(inplace=True,drop=True)
        return this_donor_proj_df
    else:
        return list(city_df[city_df['Donor ID']==DONOR_ID]['Project ID'])


# In[70]:


def get_projrecommendations_for_donor(DONOR_ID, topN,returnDetails):
    #print 'Donor ID:',DONOR_ID
    lsh_donor_id = convert_donor_id_to_lsh(DONOR_ID)
    similar_donors_of_d = get_similar_donors(lsh_donor_id,False,similar_donors)
    #print 'Similar Donors: ',similar_donors_of_d

    projects_of_this_donor = get_projects_of_this_donor(DONOR_ID,False)

    # get projects of similar donors
    recommended_projects=[]
    for similar_donor in similar_donors_of_d:
        recommended_projects.extend(get_projects_of_this_donor(similar_donor,False))

    # remove duplicated while ordering projects by decreasing frequency
    recommended_projects = map(lambda (p,c): p,               sorted(set([(proj,recommended_projects.count(proj)) for proj in recommended_projects]),                      key=operator.itemgetter(1),reverse=True))

    # remove projects donor has already donated to
    for proj in projects_of_this_donor:
        try:
            recommended_projects.remove(proj)
        except ValueError:
            pass

    #print recommended_projects[:topN]
    #print 'Number of recommendations =',len(recommended_projects[:topN])
    
    if returnDetails:
        projects_recommended_df = city_df[city_df['Project ID'].isin(recommended_projects[:topN])]    [['Project ID','Project Title','Project Type','Project Resource Category','Project Subject Category Tree','Project Subject Subcategory Tree']]
        projects_recommended_df.drop_duplicates(inplace=True)
        projects_recommended_df = projects_recommended_df.reset_index(drop=True)
        #print len(projects_recommended_df)
        return projects_recommended_df
    else:
        return recommended_projects[:topN]


# In[156]:


def compute_evaluation_metric():
    evaluate_dict={}
    for DONOR_ID in list(df['Donor ID']):
        #DONOR_ID='a5c69797ed95ffa7f18bc69e8540c676'
        #if len(evaluate_dict)==1:
        #    break
        #print DONOR_ID
        donor_projs = get_projects_of_this_donor(DONOR_ID,True)
        sim = 0  
        recommend_projs=pd.DataFrame()
        try:
            recommend_projs = get_projrecommendations_for_donor(DONOR_ID,topN,True)
        except IndexError:
            pass
        sim_proj=[0,0,0,0]
        sim_proj_t = 0
        proj_count=0
        for record_r in recommend_projs.values:
            #print record_r
            for record_d in donor_projs.values:
                if(sim_proj[0]==0 and record_d[2] == record_r[2]):
                    sim_proj[0] += 1
                if(sim_proj[1]==0 and record_d[3] == record_r[3]):
                    sim_proj[1] += 1
                if(sim_proj[2]==0 and record_d[4] == record_r[4]):
                    sim_proj[2] += 1
                if(sim_proj[3]==0 and record_d[5] == record_r[5]):
                    sim_proj[3] += 1
            sim_proj_t += sum(sim_proj)/4.0

            proj_count += 1
            sim+=sim_proj_t
            sim_proj =[0,0,0,0]
            sim_proj_t = 0
        sim += sim_proj_t
        #print sim, proj_count
        if proj_count != 0:
            sim /= proj_count
            evaluate_dict[DONOR_ID]=sim
    return sum(evaluate_dict.values())/(len(evaluate_dict.values()))


# #### Read Data

# In[55]:


t1=t()

# FILES FOR CONSTRUCTING LSH MODEL
path_processeddata = '/home/seherkhan/myfiles/coursework/usc/spring2019/inf553/proj/io/Final/lsh/Matrices_and_Data/'
proj_def_df = pd.read_csv(path_processeddata+'proj_def_oakland_1.csv').drop(['Project Resource Category.1'],axis=1) # used for recommending projects
df = pd.read_csv(path_processeddata+'ulit_mat_oakland_projtypeid_binary_1.csv')

## FILES FOR RECOMMENDATIONS
city_df = pd.read_csv(path_processeddata+'Oakland_dataset.csv')

t2=t()
print 'time taken =',t2-t1


# #### Set inputs

# In[11]:


orgM=df.iloc[:,1:]
#orgM has donors in the rows and projects in the columns

M = np.matrix(orgM.astype(int))
M.shape
# M has donors in the rows and projects in the columns
# M.shape # (1781 donors,1303 projects)


n_chunks = 10
n_projs = M.shape[1]
n_hashfunc = 20
inf = float("inf")
rows = 5 # number of rows in each band
# b = n (i.e. n_hashfunc) / rows = 20 / 5 = 4
threshold = 0.5


# #### Data Processing for LSH

# In[12]:


M_chunks = np.array_split(M, n_chunks)
# M_chunks[0].shape # shape of one chunk is approx (178 donors, 1303 projects)
M_ind_chunks=np.array_split(range(M.shape[0]),n_chunks) # used for getting recommendations

num_non_zero_Mchunks(M_chunks)


# In[13]:


# format data as {donor_id:[projects donated to]}
t1=t()
projs_of_donors_in_chunks = {}
for i in range(len(M_chunks)):
    #print i
    projs_of_donors_in_chunks.update(get_projs_of_donors_in_M_chunk(i))
t2=t()
print 'time taken =',t2-t1


# #### Run LSH

# **Compute Signatures of Each Donor**

# In[14]:


t1=t()
signatures_of_chunks = []
for i in range(len(M_chunks)):
    #print i
    signatures_of_chunks.append(compute_signature(i))
t2=t()
print 'time taken =',t2-t1

# signatures_of_chunks has length n_chunks = 10
# each value of signatures_of_chunks has n_users = 178 entries (one for each user in the chunk)
# each entry is a tuple of the form (<chunk_id>_<donor_id>, signature)
# each signature has length n_hashfunc = 20 (i,e. it is a list of 20 int values)


# In[15]:


#sc.stop()
sc = SparkContext()


# In[16]:


t1=t()
rdd = sc.parallelize(signatures_of_chunks).flatMap(lambda sigtup: sigtup)
print 'Number of donors / signatures = ',rdd.count()
print 'Number of paritions =',rdd.getNumPartitions()
# myrdd has signatures of all n_users = 6927 donors

t2=t()
print '\ntime taken =',t2-t1


# **Divide Signatures in Bands**  
# For each donor,  
#     split signature in bands of 'rows' rows each  
#     associate with each row its row_number in the band  
#     and then flatten the list of rows  
# (this gives us a list of tuples of the form (<chunk_id>\_<donor_id>,(row_id,signature_row)))

# In[17]:


t1=t()
# divide in bands
bands = rdd.mapValues(lambda sig:[sig[r:r+rows] for r in range(0,n_hashfunc,rows)]) .flatMapValues(lambda rows:[(i,rows[i]) for i in range(len(rows))])
t2=t()
print 'bands.take(2)',bands.take(2)
print '\ntime taken =',t2-t1


# **Find Candidate Pairs**  
# sort the signature_row of each tuple,  
# and then convert it to string and make (row_id,str_sorted_signature_row) the key  
# group by key to get tuples with such keys and a list of local donor_ids  
# retain only the list of local donor_ids  
# and retain only those lists of length more than 1  
# to get candidate pairs, find all possible pairs in each list, flatten the list and retain only distinct pairs
# 

# In[18]:


# find candidate pairs
t1=t()
candidate_pairs = bands.map(lambda (k,v):((v[0], str(sorted(v[1])) ),k)).groupByKey().mapValues(list).map(lambda (k,v): v).filter(lambda l:len(l)>1) .flatMap(lambda l: [v for v in combinations(l,2)]).distinct()
t2=t()
print 'candidate_pairs.take(2)',candidate_pairs.take(2)
print '\ntime taken =',t2-t1


# **Finding Similar Pairs**  
# Use projs_of_donors_in_chunks to compute the jaccard of each candidate pair  
# Note, that we used actual donor data to compute jaccard. Hence we will not have to adjust b,r and t  
# Computing jaccard this way also means that our results will contain no FPs or FNs  
#   
# A similar pair can be used to make recommendations if the jaccard is above some threshold>0

# In[19]:


# calculate jaccard of candidate pairs to find pairs to recommend
t1=t()

similar_pairs = candidate_pairs    .map(lambda pair:(pair,find_jaccard(pair)))    .filter(lambda (k,v):v>=threshold and v<=1)
print 'Number of pairs with thershold<=jaccard<=1  =',similar_pairs.count()
print

t2=t()
print 'time taken =',t2-t1


# In[20]:


t1=t()
# has jaccard
similar_donors0 = similar_pairs.flatMap(lambda (k,v):[(k[0],(k[1],v)),(k[1],(k[0],v))])            .groupByKey().mapValues(lambda vals: sorted(vals,key=operator.itemgetter(1),reverse=True))
# without jaccard
similar_donors = similar_donors0.mapValues(lambda vals: [k for (k,v) in vals]).sortByKey()
print 'Number of donors that have a pair with threshold < jaccard <= 1 =', similar_donors.count()

t2=t()
print 'time taken =',t2-t1


# ### Making Recommendations given a Donor ID

# In[133]:


#lsh_donor_id = '6_85'
#DONOR_ID = convert_donor_id_to_org(lsh_donor_id)
DONOR_ID =  'a5c69797ed95ffa7f18bc69e8540c676' #'02af341a795653432a8a3e3d5968cd30'

topN=10

get_projrecommendations_for_donor(DONOR_ID,topN,True)


# In[134]:


get_projects_of_this_donor(DONOR_ID,True)


# #### Compute Evaluation Metric

# In[158]:


t1=t()
print compute_evaluation_metric()
t2=t()

print 'time taken =',t2-t1


# In[39]:


sc.stop()

