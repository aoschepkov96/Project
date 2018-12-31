#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd ## Обновить pandas до последней версии
import numpy as np
import math
import os
import glob
import mysql.connector
import datetime
from mysql.connector import errorcode
from sklearn import preprocessing
from sklearn import metrics

import matplotlib.pyplot as plt
import pandas.core.algorithms as algos
from pandas import Series
import scipy.stats.stats as stats
import re
import traceback
from datetime import date
import string
from IPython.display import display, HTML

from os import listdir
from os.path import isfile, join


import warnings
warnings.filterwarnings("ignore")


# In[2]:


my_path = input("Введите путь папки, в которой будут лежать все ваши файлы (пример формата C:\\Users\\User\\Documents\\Python Scripts\\IT\\Untitled Folder): ")
while my_path  == '':
    print("Ошибка: вы не ввели путь папки")
    my_path = input("Введите путь папки, в которой будут лежать все ваши файлы (пример формата C:\\Users\\User\\Documents\\Python Scripts\\IT\\Untitled Folder): ")
print("Вы ввели '" + str(my_path)+"'")

print("Введите данные для использования MySQL")
user = input("Введите имя пользователя (user name): ")
while user  == '':
    print("Ошибка: вы не ввели имя пользователя")
    user = input("Введите имя пользователя (user name): ")
print("Вы ввели '" + str(user)+"'")
database = input("Введите название базы данных (database name): ")
while database  == '':
    print("Ошибка: вы не ввели название базы данных")
    database = input("Введите название базы данных (database name): ")
print("Вы ввели '" + str(database)+"'")


# #### Блок Калинина Артёма

# In[3]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

payments_table = pd.read_sql_query("SELECT * FROM payments", cnx)
payments_table.keys()


# In[4]:


def get_90_day_default(contract_number):
    table = payments_table[payments_table['contract_number'] == contract_number]## Выбираем строки с конкретным номером контракта
    table = table.reset_index() #обнуляем индексы, чтобы при дальнейшей итерации всё работало с 0
    for i in  range(0, len(table['amount_paid'])):
        try:
            if (table['amount_paid'][i] < table['amount_due'][i] and table['amount_paid'][i+1] < table['amount_due'][i+1] and
                table['amount_paid'][i+2] < table['amount_due'][i+2] and table['amount_paid'][i+3] < table['amount_due'][i+3]):
                #Проверяем чтобы заёмщик не платил 4 даты подряд
                date_90 = table['payment_date'][i + 3] #фиксируем дату просрочки
                date_90 = (str(pd.to_datetime(date_90).year) + '-' + str(pd.to_datetime(date_90).month) + 
                           '-' + str(pd.to_datetime(date_90).day)) #переводим в нужный формат
                break
                
            else:
                date_90 = '0000-00-00' #если нет просрочки то передаём такую дату
        except:
            True
    return date_90


# In[5]:


data = []
for i in payments_table['contract_number'].unique():
    if get_90_day_default(i) == '0000-00-00':
        data.append((i, get_90_day_default(i), 0))
    else:
        data.append((i, get_90_day_default(i), 1))
data


# In[13]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

cursor.execute("""ALTER TABLE contracts ADD default_ INT """)

for i in data:
    cursor.execute("""UPDATE contracts SET default_ = %d WHERE contract_number = %d """ %(i[2], i[0]))

#'0000-00-00' Означает что нет просрочки

cnx.commit()


# In[14]:


applications_defaults_and_id = pd.read_sql_query("""SELECT contracts.id, age, car_owner, children, education, employed_by, family, 
gender, house_ownership, housing, income, income_type, marital_status, position, default_
FROM applications
INNER JOIN contracts ON
contracts.id = applications.id;""", cnx)
applications_defaults_and_id.head()


# In[15]:


defaults = applications_defaults_and_id.default_
defaults


# In[41]:





# In[16]:


applications_defaults = applications_defaults_and_id.drop(['default_', 'id'], axis = 1)


# ### Правила обработки информации:
#  1. Каждой категории текстовых полей будет присваиваться балл
#  2. Непрерывные переменные сначала группируются по 25% выборки

# In[17]:


#defaults = applications_defaults.default_
#defaults


# In[19]:


max_bin = 20
force_bin = 4

# define a binning function
def char_bin(Y, X):
        
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X','Y']][df1.X.isnull()]
    notmiss = df1[['X','Y']][df1.X.notnull()]    
    df2 = notmiss.groupby('X',as_index=True)
    
    d3 = pd.DataFrame({},index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y
    
    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan},index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4,ignore_index=True)
    
    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = d3.EVENT/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = d3.NONEVENT/d3.sum().NONEVENT
    d3["WOE"] = np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME','MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE', 'DIST_EVENT','DIST_NON_EVENT','WOE', 'IV']]      
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    
    return(d3)

def woe_iv_categorical(df1, target):
    
    stack = traceback.extract_stack()
    filename, lineno, function_name, code = stack[-2]
    vars_name = re.compile(r'\((.*?)\).*$').search(code).groups()[0]
    final = (re.findall(r"[\w']+", vars_name))[-1]
    
    x = df1.dtypes.index
    count = -1
    
    for i in x:
        if i.upper() not in (final.upper()):
            conv = char_bin(target, df1[i])
            conv["VAR_NAME"] = i
            count = count + 1
            if count == 0:
                iv_df = conv
            else:
                iv_df = iv_df.append(conv,ignore_index=True)
    
    iv = pd.DataFrame({'IV':iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return(iv_df,iv)


# In[20]:


applications_defaults_1  = applications_defaults.copy()


# In[21]:


applications_defaults_1['age'].describe()


# In[22]:


applications_defaults_1['car_owner'].describe()


# In[23]:


applications_defaults_1['income'].describe()


# In[24]:


applications_defaults_1['age'] = np.where( (applications_defaults_1['age'] <= 34) & (applications_defaults_1['age'] >= 21), 0, 
                               np.where ((applications_defaults_1['age'] <= 46) & (applications_defaults_1['age'] > 34), 1,
                                        np.where ((applications_defaults_1['age'] <= 54) & (applications_defaults_1['age'] > 46), 2, 3)))


# In[25]:


applications_defaults_1['income'] = np.where( (applications_defaults_1['income'] <= 96750) & (applications_defaults_1['income'] >= 38419), 0, 
                               np.where ((applications_defaults_1['income'] <= 126000) & (applications_defaults_1['income'] > 96750), 1,
                                        np.where ((applications_defaults_1['income'] <= 202500) & (applications_defaults_1['income'] > 126000), 2, 3)))


# In[27]:


applications_defaults_1.head(5)


# In[ ]:





# In[28]:


woe_iv_cat_df = woe_iv_categorical(applications_defaults_1, defaults)[0]
print('Полученная таблица WOE и IV по всем переменным:')
display(woe_iv_cat_df)


# ### Правило присвоения баллов: 
#  1. Ранжируем категории в каждой переменной по возрастанию WOE
#  2. Самому низкому WOE назначается балл равный одному - далее по возрастанию WOE каждой категории добавляется три балла
#  3. Полученные таким образом баллы нормализуются к шкале от 1 до 100: количество баллов конкретной категории делится на сумму уникальных баллов и умножается на 100%

# In[29]:


def get_scores(var_name):
    count = 0
    values = [0]
    for i in range(0, woe_iv_cat_df[woe_iv_cat_df['VAR_NAME'] == var_name].sort_values(['WOE']).shape[0] - 1):
        table = woe_iv_cat_df[woe_iv_cat_df['VAR_NAME'] == var_name].sort_values(['WOE'])
        if table.iloc[i][10] == table.iloc[i + 1][10]:
            values.append(count)
        else:
            count = count + 3
            values.append(count)
    c = table
    c['scores'] = values
    return c


# In[30]:


new_table = pd.DataFrame()
for i in woe_iv_cat_df['VAR_NAME'].unique():
    new_table = new_table.append(get_scores(i))


# In[31]:


our_dict = []
for i in new_table['VAR_NAME'].unique():
    old_keys = new_table[new_table['VAR_NAME'] == i].MIN_VALUE
    new_scores = new_table[new_table['VAR_NAME'] == i].scores
    dictionary = dict(zip(old_keys, new_scores))
    our_dict.append((i, dictionary))


# In[32]:


our_dict


# In[33]:


for i in our_dict:
    applications_defaults_1[i[0]] = applications_defaults_1[i[0]].map(i[1] )
applications_defaults_1.head()


# In[34]:


for i in applications_defaults_1:
    applications_defaults_1[i] = [(x / applications_defaults_1[i].unique().sum() * 100) for x in applications_defaults_1[i]]


# In[35]:


applications_defaults_1


# In[36]:


applications_defaults_1['id'] = applications_defaults_and_id['id']
cols = list(applications_defaults_1)
cols.insert(0, cols.pop(cols.index('id')))
applications_defaults_1 = applications_defaults_1.ix[:, cols]
print('Полученная таблица со скорингами по каждой переменной для каждого заемщика:')
display(applications_defaults_1)


# In[37]:


cnx.close()


# ### Выгрузка из этого блока

# In[48]:


applications_defaults_1.to_csv(my_path + '\\output_3\\scores_for_vars.csv', header=True)
woe_iv_cat_df.to_csv(my_path + '\\output_3\\woe_iv_table.csv', header=True)


# In[51]:


applications_defaults_and_id[['id', 'default_']].to_csv(my_path + '\\output_3\\defaults_storage.csv', header=True)


# In[46]:





# In[1]:


print("""part_3 выполнена, расчитаны WOE и IF по всем переменным, рассчитаны кредитные скоры заёмщиков по каждой переменной
посмотрите на результаты в папке output_3""")


# In[ ]:




