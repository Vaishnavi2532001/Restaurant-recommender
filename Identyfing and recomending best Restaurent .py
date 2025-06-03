#!/usr/bin/env python
# coding: utf-8

# In[94]:


## importing Liabreries 

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns 
import statistics as stc 


# In[95]:


data = pd.read_excel('data.xlsx')


# In[9]:


pd.read_excel('data.xlsx')
data.head(10)


# In[10]:


cc = pd.read_excel('Country-Code.xlsx')


# In[11]:


cc


# In[12]:


merged = pd.merge(data, cc,on='Country Code', how = 'left')
merged.head()


# In[14]:


merged.info()


# In[15]:


merged.isna().sum()
#merged.isnull().sum() #total number of null entries per column


# In[16]:


merged.dropna(axis=0,subset=['Restaurant Name'],inplace=True)


# In[17]:


merged[merged['Cuisines'].isnull()]


# In[20]:


#Since there were only 9 records without cuisines, we have replace the null␣
#values with Others.
merged['Cuisines'].fillna('Others',inplace=True)


# In[21]:


#duplicate data finding
duplicateRowsDF = merged[merged.duplicated()]
print("Duplicate Rows except first occurrence based on all columns are :")
print(duplicateRowsDF)


# # EDA 1

# In[24]:


country_distri = merged.groupby(['Country Code','Country']).agg(Count=('Restaurant ID','count'))
country_distri.sort_values(by='Count',ascending=False)


# In[25]:


country_distri.plot(kind= 'barh')


# In[27]:


city_dist = merged.groupby(['Country','City']).agg(Count=('Restaurant ID','count'))
city_dist.describe()


# In[28]:


city_dist.sort_values(by='Count',ascending=False)
# New Delhi has maximum number of restaurant


# In[29]:


# Minimum number of restaurant in following cities
min_cnt_rest = city_dist[city_dist['Count']==1]
min_cnt_rest.info()
min_cnt_rest


# In[31]:


# Find out the ratio between restaurants that allow table booking vs. those␣
#,→that do not allow table booking
merged1 = merged.copy()
merged1.columns


# In[32]:


dummy = ['Has Table booking','Has Online delivery']
merged1 = pd.get_dummies(merged1,columns=dummy,drop_first=True)
merged1.head()
# 0 indicates 'NO'
# 1 indicates 'YES'


# In[35]:



#Ration between restaurants allowing table booking and those which dont
tbl_book_y = merged1[merged1['Has Table booking_Yes']==1]['Restaurant ID'].count()
tbl_book_n = merged1[merged1['Has Table booking_Yes']==0]['Restaurant ID'].count()
print('Ratio between restaurants that allow table booking vs. those that do not␣allow table booking: ',
round((tbl_book_y/tbl_book_n),2))


# In[37]:


#Pie chart to show percentage of restaurants which allow table booking and␣
("#→those", "which", "don't")
labels = 'Table Booking', 'No Table Booking'
sizes = [tbl_book_y,tbl_book_n]
explode = (0.1, 0) # only "explode" the 2nd slice (i.e. 'Hogs')
fig1, ax1 = plt.subplots(figsize=(5,5))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
ax1.set_title("Table Booking vs No Table Booking")
ax1.axis('equal') # Equal aspect ratio ensures that pie is drawn as a circle.
plt.show()


# In[43]:


# Find out the percentage of restaurants providing online delivery

order_on = merged1[merged1['Has Online delivery_Yes'] == 1]['Restaurant ID'].count()
order_off = merged1[merged1['Has Online delivery_Yes'] == 0]['Restaurant ID'].count()
print('Percentage of restaurants providing online delivery : {} %'.
format((round(order_on/len(merged1),3)*100)))


# In[45]:


#pie chart to show percentages of restaurants allowing online delivery vs those␣
# which do not have online delivery

labels = 'Online Delivery','No Online Delivery'
size = [order_on,order_off]
explode = (0.1,0)
fig1,ax1 = plt.subplots(figsize=(5,5))
ax1.pie(size,explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)
ax1.set_title("Online Delivery vs No Online Delivery")
ax1.axis('equal')
plt.show()


# In[46]:


sns.boxplot(merged['Votes'])


# In[48]:


# Calculate the difference in number of votes for the restaurants that deliver␣
#and the restaurants that do not deliver
# first detect and remove (replace it with mean/closest possible value) outlier␣
#for VOTE

import sklearn
import pandas as pd
''' Detection '''
# IQR
Q1 = np.percentile(merged['Votes'], 25,
interpolation = 'midpoint')
Q3 = np.percentile(merged['Votes'], 75,
interpolation = 'midpoint')
IQR = Q3 - Q1
print("Old Shape: ", merged.shape)
# Upper bound
upper = np.where(merged['Votes'] >= (Q3+1.5*IQR))
#print("Upper bound:",upper)
#print(np.where(upper))
# Lower bound
lower = np.where(merged['Votes'] <= (Q1-1.5*IQR))
''' Removing the Outliers '''
merged.drop(upper[0], inplace = True)
merged.drop(lower[0], inplace = True)
print("New Shape: ", merged.shape)


# In[49]:


sns.boxplot(merged['Average Cost for two'])


# In[54]:


import sklearn
import pandas as pd

''' Detection '''

# IQR

Q1 = np.percentile(merged['Average_Cost_for_two'], 25,interpolation = 'midpoint')

Q3 = np.percentile(merged['Average_Cost_for_two'], 75, interpolation = 'midpoint')
IQR = Q3 - Q1


print("Old Shape: ", merged.shape)

# Upper bound

upper = np.where(merged['Average_Cost_for_two'] >= (Q3+1.5*IQR))
print("Upper bound:",upper)
print(np.where(upper))

# Lower bound

lower = np.where(merged['Average_Cost_for_two'] <= (Q1-1.5*IQR))

''' Removing the Outliers '''
merged.drop(upper[0], inplace = True)
merged.drop(lower[0], inplace = True)


# In[52]:


print("New Shape: ", merged.shape)


# In[55]:


import sklearn
import pandas as pd 

# IQR 
Q1= np.percentile(merged['Average_Cost_for_two'],25,
                 interpolation='Midnight')

Q2= np.percentile(merged['Average_Coast_for_two'],75,
                 interpolation='midpoint')
IQR = Q2-Q1

print("old shape: ",merged.shape)

#upper bound 
upper= np.where(merged['Average_cost_for_two']>=(Q3+1.5*IQR))
print("upper bond:",upper)
print(np.where(upper))


# In[56]:


print("New Shape: ", merged.shape)


# In[57]:


merged.hist(['Votes', 'Average Cost for two'], figsize=(18,5))


# In[58]:


dimen=(18,5)
fig, ax = plt.subplots(figsize=dimen)
sns.boxplot(x='Votes',y='Has Online delivery',data=merged,ax=ax)


# In[59]:


rest_deliver = merged1[merged1['Has Table booking_Yes'] == 1]['Votes'].sum()
rest_ndeliver = merged1[merged1['Has Table booking_Yes'] == 0]['Votes'].sum()
print('Difference in number of votes for restaurants that deliver and dont deliver: ',abs((rest_deliver - rest_ndeliver)))


# In[61]:


labels = 'Online Delivery','No Online Delivery'
size = [rest_ndeliver,rest_deliver]
explode = (0,0.1)
fig1,ax1 = plt.subplots(figsize=(5,5))
ax1.pie(size,explode=explode,labels=labels,autopct='%1.1f%%',shadow=True,startangle=90)
ax1.set_title("Votes: Online Delivery vs Votes:No Online Delivery")
ax1.axis('equal')
plt.show()


# In[62]:


#out of the total votes about 27.3% votes were given to restaurants that dont␣
#,→have online delivery option
#out of the total votes about 72.7% votes were given to restaurants that dont␣
#,→have online delivery option
#This clearly shows that restaurants that have online delivery are more likely␣
#,→to get a vote(feedback)


# In[63]:


# What are the top 10 cuisines served across cities?
top_10_couisines = merged.groupby(['City','Cuisines']).agg( Count =('Cuisines','count'))
df=top_10_couisines.sort_values(by='Count',ascending=False)
#top_10_couisines = merged['Cuisines'].value_counts()
#top_10_couisines.head(10)
#top_10_couisines.plot(kind='barh')
df.head(10).plot(kind='bar')


# In[65]:


# What is the maximum and minimum number of cuisines that a restaurant serves?
cuis_count = merged.groupby(['Restaurant Name','Cuisines']).agg( Count =('Cuisines','count'))
cuis_count.sort_values(by='Count',ascending=False)


# In[66]:


# Also, which is the most served cuisine across the restaurant for each city?
cuis_count_ct = merged.groupby(['City','Cuisines']).agg( Count =('Cuisines','count'))
cuis_count_ct.sort_values(by='Count',ascending=False)


# In[67]:


merged.columns


# In[68]:


cuisines = merged['Cuisines'].apply(lambda x: pd.Series(x.split(',')))
cuisines


# In[69]:


cuisines.columns =['Cuisine_1','Cuisine_2','Cuisine_3','Cuisine_4','Cuisine_5','Cuisine_6','Cuisine_7','Cuisine_8']
cuisines.tail()


# In[70]:


df_cuisines = pd.concat([merged,cuisines],axis=1)
df_cuisines.head()


# In[71]:


cuisine_loc = pd.DataFrame(df_cuisines[['Country','City','Locality Verbose','Cuisine_1','Cuisine_2','Cuisine_3',
'Cuisine_4','Cuisine_5','Cuisine_6','Cuisine_7','Cuisine_8']])


# In[72]:


cuisine_loc_stack=pd.DataFrame(cuisine_loc.stack()) #stacking the columns
cuisine_loc.head()


# In[73]:


cuisine_loc_stack.head(10)


# In[75]:


keys = [c for c in cuisine_loc if c.startswith('Cuisine')]
a=pd.melt(cuisine_loc, id_vars='Locality Verbose', value_vars=keys,value_name='Cuisines')

#melting the stack into one row


# In[76]:


max_rate=pd.DataFrame(a.groupby(by=['Locality Verbose','variable','Cuisines']).size().reset_index())
#find the highest restuarant in the city
max_rate
del max_rate['variable']
max_rate.columns=['Locality Verbose','Cuisines','Count']
max_rate


# In[77]:


loc=max_rate.sort_values('Count', ascending=False).groupby(by=['Locality Verbose'],as_index=False).first()
loc


# In[78]:


rating_res=loc.merge(merged1,left_on='Locality Verbose',right_on='Locality Verbose',how='inner')
#inner join to merge the two dataframe
rating_res


# In[79]:


df=pd.DataFrame(rating_res[['Country','City','Locality Verbose','Cuisines_x','Count']])
#making a dataframe of rating restaurant
df


# In[80]:


country=rating_res.sort_values('Count', ascending=False).groupby(by=['Country'],as_index=False).first()
#grouping the data by country code
country


# In[81]:


con=pd.DataFrame(country[['Country','City','Locality','Cuisines_x','Count']])
con.columns=['Country','City','Locality','Cuisines','Number of restaurants in the country']
#renaming the columns
con


# In[82]:


con1=con.sort_values('Number of restaurants in the country', ascending=False)
#sorting the restaurants on the basis of the number of restaurants in the country
con1[:10]


# In[83]:


import matplotlib.pyplot as plt

plt.bar(con1['Cuisines'],con1['Number of restaurants in the country'])
plt.xlabel("Cuisines")
plt.ylabel("Number of restaurants in the country")
plt.xticks(rotation=90)

#con1.plot(kind='bar')


# In[84]:


rest_cuisine = pd.DataFrame(df_cuisines[['Restaurant Name','City','Cuisine_1','Cuisine_2','Cuisine_3','Cuisine_4',
'Cuisine_5','Cuisine_6','Cuisine_7','Cuisine_8']])
rest_cuisine_stack=pd.DataFrame(rest_cuisine.stack()) #stacking the columns
rest_cuisine.head()


# In[85]:


keys1 = [c for c in rest_cuisine if c.startswith('Cuisine')]
b=pd.melt(rest_cuisine, id_vars='Restaurant Name', value_vars=keys, value_name='Cuisines')
#melting the stack into one row
max_rate1=pd.DataFrame(b.groupby(by=['Restaurant Name','variable','Cuisines']).size().reset_index())
#find the highest restuarant in the city
max_rate1
del max_rate1['variable']
max_rate1.columns=['Restaurant Name','Cuisines','Count']
max_rate1.head(10)


# In[86]:


max_rate1.sort_values('Count',ascending=False)
#Cafe Coffee Day has the max number of cuisines and The least number of cuisines in a resaurant is 1.


# In[88]:


rating = merged1[['Restaurant ID','Restaurant Name','Country','City','Aggregate rating','Average Cost for two','Votes','Price range','Has Table booking_Yes','Has Online delivery_Yes']]


# In[89]:


rating = rating.merge(max_rate1,left_on='Restaurant Name',right_on='Restaurant Name',how='left')
rating


# In[90]:


merged1.corr()


# In[91]:


fig, ax = plt.subplots(figsize=(18,8))
dataplot = sns.heatmap(merged1.corr(), cmap="YlGnBu", annot=True,linewidth=0.5,ax=ax)
#heat = merged1.pivot("Average_Cost_for_two", "Aggregate_rating")
#ax = sns.heatmap(heat, annot=True, fmt="d")


# In[93]:


#We see that there is no single variable that affects the rating strongly, however table booking,online
#delivery,avg price for two and price range, number of votes do play a part in affecting the rating of
#a restaurant.


# In[ ]:




