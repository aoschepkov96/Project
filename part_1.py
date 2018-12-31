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


# #### Блок Ивахненко Анастасии

# In[3]:


applications = [f for f in listdir(my_path+'/applications') if isfile(join(my_path+'/applications', f))] 
contracts = [f for f in listdir(my_path+'/contracts') if isfile(join(my_path+'/contracts', f))]


# In[4]:


df = pd.read_excel(my_path+'/applications/' + applications[0], header = None, usecols = [0, 3, 4, 5, 6, 7, 8, 12, 14]) 
#Читаем первый excel, чтобы получить названия колонок


# In[5]:


columns = [df[0][4], df[0][13], df[0][15], df[0][17], df[0][19], df[1][4], df[2][6], df[2][17], df[3][6], df[4][2], df[4][6],
          df[5][2], df[6][2], df[6][4], df[7][8], 'application_date', df[8][2]]
#список с названиями колнок


# In[6]:


values = []
for i in applications:
    df = pd.read_excel(my_path+'/applications/' + i, header = None, usecols = [0, 3, 4, 5, 6, 7, 8, 12, 14])
    if df.shape[0] < 21:
        df.loc[20] = np.nan
    values.append((df[0][5], df[0][14], df[0][16], df[0][18], df[0][20], df[1][5], df[2][7], df[2][18], 
                   df[3][7], df[4][3], df[4][7], df[5][3], df[6][3], df[6][5], df[7][9], df[8][1], df[8][3]))


# In[7]:


applic_df = pd.DataFrame(data = values, columns = columns)
applic_df.head(4)


# In[8]:


applic_df = applic_df.drop(['Date appointed'], axis = 1)
applic_df = applic_df.drop(['Issue Date'], axis = 1)


# In[9]:


def calculate_age(born):
    today = date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


# In[10]:


ages = []
for i, value in enumerate(applic_df['Date of Birth']):
    try:
        ages.append(calculate_age(datetime.datetime.strptime(applic_df['Date of Birth'][i], '%m.%d.%Y')))
    except:
        ages.append(calculate_age(value))
        
applic_df['age'] = ages


# In[11]:


for i, value in enumerate(applic_df['application_date']):
    try:
        applic_df['application_date'].loc[i] = datetime.date.strftime(datetime.datetime.strptime(value, '%m.%d.%Y'), "%Y-%m-%d")
    except:
        applic_df['application_date'].loc[i] = (str(applic_df['application_date'][i].year) + 
                                                '-' + str(applic_df['application_date'][i].month) + '-' + 
                                                str(applic_df['application_date'][i].day))


# In[12]:


applic_df = applic_df.where((pd.notnull(applic_df)), None)


# In[13]:


applic_df = applic_df.drop(['Date of Birth'], axis = 1)


# In[14]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

cursor.execute("""CREATE TABLE IF NOT EXISTS applications 
(id INT, 
income INT, 
income_type VARCHAR(30), 
housing VARCHAR (30), 
age_of_car INT,  
children INT,
house_ownership VARCHAR(30), 
family INT,
marital_status VARCHAR(30), 
gender VARCHAR(30),
employed_by VARCHAR(30), 
education VARCHAR(30),
position VARCHAR(30),
application_date DATE,
age INT,
PRIMARY KEY (id)) ENGINE=INNODB""")


# In[15]:


# заполняем значениями таблицу applications
sql = "INSERT INTO applications VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
for i in range(0, applic_df.shape[0]):
    val = (int(applic_df.iloc[i]['Identity Number']), applic_df.iloc[i]['Income'], applic_df.iloc[i]['Income Type'],
           applic_df.iloc[i]['Housing'], applic_df.iloc[i]['Age of Car (if owned)'], applic_df.iloc[i]['Children'], 
           applic_df.iloc[i]['House ownership'], applic_df.iloc[i]['Family'], applic_df.iloc[i]['Marital Status'], 
           applic_df.iloc[i]['Gender'], applic_df.iloc[i]['Employed By'], applic_df.iloc[i]['Education'], 
           applic_df.iloc[i]['Position'], applic_df.iloc[i]['application_date'], int(applic_df.iloc[i]['age']))
    cursor.execute(sql, val)

    cnx.commit()


# ## Contracts

# In[16]:


df_contract = pd.read_excel(my_path + '/contracts/' + contracts[0], header = None, usecols = [0, 4, 5])


# In[17]:


column_values_contracts = [df_contract[0][4], df_contract[0][6], df_contract[0][8], 
                           df_contract[1][4], df_contract[1][6], df_contract[1][8], 'contract_date']


# In[18]:


values_contracts = []
for i in contracts:
    df = pd.read_excel(my_path + '/contracts/' + i, header = None, usecols = [0, 4, 5])
    values_contracts.append((df[0][5], df[0][7], df[0][9], df[1][5], df[1][7], df[1][9], df[2][1]))


# In[19]:


contract_df = pd.DataFrame(data = values_contracts, columns = column_values_contracts)
contract_df.head(5)


# In[20]:


for i, value in enumerate(contract_df['contract_date']):
    try:
        contract_df['contract_date'].loc[i] = datetime.date.strftime(datetime.datetime.strptime(value, '%m.%d.%Y'), "%Y-%m-%d")
    except:
        contract_df['contract_date'].loc[i] = (str(contract_df['contract_date'][i].year) + 
                                                '-' + str(contract_df['contract_date'][i].month) + '-' + 
                                                str(contract_df['contract_date'][i].day))


# In[21]:


contract_df = contract_df.where((pd.notnull(contract_df)), None)


# In[22]:


cnx.close()


# In[23]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

cursor.execute("""CREATE TABLE IF NOT EXISTS contracts 
(id INT, 
amount INT, 
term_month INT, 
contract_number INT, 
type VARCHAR(30), 
annuity INT, 
contract_date DATE,
PRIMARY KEY (contract_number),
FOREIGN KEY (id) REFERENCES applications(id)) ENGINE=INNODB """)


# In[24]:


sql = "INSERT INTO contracts VALUES (%s, %s, %s, %s, %s, %s, %s)"
for i in range(0, contract_df.shape[0]):
    val = (int(contract_df.iloc[i]['Identity Number']), contract_df.iloc[i]['Amount'], int(contract_df.iloc[i]['Term (month)']),
           int(contract_df.iloc[i]['Contract Number']), contract_df.iloc[i]['Type'], contract_df.iloc[i]['Annuity'], 
           contract_df.iloc[i]['contract_date'])
    try:
        cursor.execute(sql, val)
    except:
        print("Нет заявки с id = %d, такой контракт не передаётся в базу" %contract_df['Identity Number'][i])

    cnx.commit()


# ## Payments

# In[92]:


payments = pd.read_excel(my_path+"/payments.xls")
payments.head(5)


# In[93]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

cursor.execute("""CREATE TABLE IF NOT EXISTS payments 
(contract_number INT, 
payment_date DATE, 
amount_due DECIMAL(18,2), 
amount_paid DECIMAL(18,2),
FOREIGN KEY (contract_number) REFERENCES contracts(contract_number))""")


# In[94]:


sql = "INSERT INTO payments VALUES (%s, %s, %s, %s)"
for i in range(0, payments.shape[0]):
    val = (int(payments.iloc[i]['Contract Number']), payments.iloc[i]['Date'], payments.iloc[i]['Amount Due'],
                payments.iloc[i]['Amount Paid'])
    try:
        cursor.execute(sql, val)
    except:
        print("Нет контракта с contract_number = %d, такой платёж не передаётся в базу" %payments['Contract Number'][i])
cnx.commit()


# In[ ]:





# In[ ]:





# ### Пункт 2. Проверка на корректность значений, исправление некоторых ошибок

# In[95]:


# позволяет посмотреть для всех текстовых переменных частоту появлений каждой категории,
# для каждой переменной формата integer или дата посмотреть на максимальное, минимальное значение и количество пропусков
def value_check (table, data_type):
    cursor.execute("""SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
    WHERE DATA_TYPE = '{0}' AND TABLE_SCHEMA = '{1}'
    AND TABLE_NAME = '{2}'""".format(data_type, database, table))
    data_type_col = pd.DataFrame(cursor.fetchall())[0].tolist()
    for i in range(0,len(data_type_col)):
        if data_type == 'varchar':
            cursor.execute("""SELECT {0}, COUNT({0}) AS count FROM {1} 
            WHERE {0} IS NOT NULL GROUP BY {0} UNION ALL SELECT {0}, 
            COUNT(CASE WHEN {0} IS NULL THEN 1 END) AS count FROM {1} 
            WHERE {0} IS NULL GROUP BY {0}""".format(data_type_col[i], table))
        if data_type == 'int' or data_type == 'date':
            cursor.execute("""SELECT MAX({0}), MIN({0}), 
            COUNT(CASE WHEN {0} IS NULL THEN 1 END) AS null_count FROM {1}""".format(data_type_col[i], table))
        data = pd.DataFrame(cursor.fetchall())
        data.columns = [x[0] for x in cursor.description]
        print ("Проверка обозначений по полю "+data_type_col[i])
        display(data)
        print("- "*30)
        print()


# In[96]:


# проверяет, есть ли в числовых данных отрицательные значения (в нашем случае для всех полей такое значение некорректно) 
# и, если есть, заменяет все отрицательные числовые значения на NULL
def int_negative_value_update (table):
    cursor.execute("""SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
    WHERE DATA_TYPE = 'int' AND TABLE_SCHEMA = '{0}'
    AND TABLE_NAME = '{1}'""".format(database, table))
    data_type_col = pd.DataFrame(cursor.fetchall())[0].tolist()
    for i in range(0,len(data_type_col)):
        cursor.execute("""SELECT COUNT(CASE WHEN {0}<0 THEN 1 END) FROM {1}""".format(data_type_col[i], table)) 
        data = cursor.fetchall()[0][0]
        if data != 0:
            cursor.execute("""UPDATE {0} SET {1}= NULL WHERE {1}<0""".format(table, data_type_col[i]))
            cnx.commit()
            print ("Отрицательные значения найдены для поля "+ data_type_col[i] + ", некорректные значения заменены на None")
        else:
            print ("Отрицательные значения для поля "+ data_type_col[i] + " не найдены, ошибок нет")


# In[97]:


# поиск и исправление ошибок в датах
def date_value_update (table):
    cursor.execute("""SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
    WHERE DATA_TYPE = 'date' AND TABLE_SCHEMA = '{0}'
    AND TABLE_NAME = '{1}'""".format(database, table))
    data_type_col = pd.DataFrame(cursor.fetchall())[0].tolist()
    #проверка на даты из будущего
    for i in range(0,len(data_type_col)):
        cursor.execute("""SELECT COUNT(CASE WHEN {0}>'2018-12-31' THEN 1 END) FROM {1}""".format(data_type_col[i], table)) 
        data = cursor.fetchall()[0][0]
        if data != 0:
            cursor.execute("""UPDATE {0} SET {1}= NULL WHERE {1}>'2018-12-31'""".format(table, data_type_col[i]))
            cnx.commit()
            print ("Среди дат по полю "+ data_type_col[i] + " найдены будущие значения, некорректные значения заменены на None")
        else:
            print ("Среди дат по полю "+ data_type_col[i] + " некорректные значения не найдены")
    print("- "*30)
    #проверка возраста до 18 лет
    cursor.execute("""SELECT id, age FROM applications WHERE age < 18""")
    data = cursor.fetchall()
    if not data:
        print("Заёмщиков младше 18 лет не найдено")
    else:
        data1 = pd.DataFrame(data)
        data1.columns = [x[0] for x in cursor.description]
        print("Найдены следующие заёмщики младше 18 лет:")
        display(data1)


# #### Проверка значений в таблице applications

# In[98]:


table = 'applications'


# In[99]:


print('Проверим таблицу заявок по всем категориальным полям:')


# In[100]:


data_type = 'varchar'
value_check(table, data_type)


# In[101]:


print('Как мы видим, нарушения в обозначениях не наблюдаются нигде, кроме поля position, поскольку в нём присутствует значение undefined. Заменим данное значение в таблице на стандартное для пропуска None.')


# In[102]:


cursor.execute("UPDATE applications SET position = NULL WHERE position = '<undefined>'")
cursor.execute("""SELECT ROW_COUNT()""")
counter = cursor.fetchall()[0][0]
cnx.commit()
print(counter, " значений <undefined> заменены на None")


# Проверка на соотношение количества членов семьи и детей (количество членов семьи не должно превышать сумму детей и родителей в зависимости от семейного положения родителей)

# In[103]:


cursor.execute("""SELECT id, family, children, marital_status FROM applications 
WHERE marital_status = 'Married' OR marital_status = 'Civil marriage' GROUP BY id
HAVING COUNT(CASE WHEN family <> children + 2 THEN 1 END) UNION ALL SELECT id, family, children, marital_status FROM applications 
WHERE marital_status = 'Separated' OR marital_status = 'Widow' OR marital_status = 'Single / not married' GROUP BY id
HAVING COUNT(CASE WHEN family <> children + 1 THEN 1 END)""")
data = cursor.fetchall()
if not data:
    print('Ошибки в соотношении количества членов семьи и детей не найдены')
else:
    data1 = pd.DataFrame(data)
    data1.columns = [x[0] for x in cursor.description]
    print('Ошибки в соотношении количества членов семьи и детей найдены по следующим заёмщикам:')
    display(data1)


# In[104]:


print('Проверка числовых переменных:')


# In[105]:


data_type = 'int'
value_check(table, data_type)


# Если в данных есть отрицательные значения, исправим это, если нет - укажем, что ошибка не обнаружена

# In[106]:


print('Обнаружили очень много пустых значения для поля age_of_car, заменим его на более информативные категории есть машина - нет машины')


# In[107]:


cursor.execute("ALTER TABLE applications ADD car_owner VARCHAR(30)")
cursor.execute("UPDATE applications SET car_owner = 'no_car' WHERE age_of_car is NULL")
cnx.commit()
cursor.execute("UPDATE applications SET car_owner = 'owns_car' WHERE age_of_car is not NULL")
cnx.commit()
cursor.execute("ALTER TABLE applications DROP age_of_car")


# In[108]:


int_negative_value_update('applications')


# Проверим значения в полях с датами:

# In[109]:


data_type = 'date'
value_check(table, data_type)


# Проверим, есть ли в данных с датами некорректные значения (даты из будущего, займщики младше 18 лет), создадим поле возраста

# In[110]:


date_value_update(table)


# #### Проверка значений в таблице контрактов

# In[111]:


table = 'contracts'


# Проверим таблицу контрактов по всем категориальным полям

# In[112]:


data_type = 'varchar'
value_check(table, data_type)


# Как мы видим, нарушения в обозначениях не наблюдаются

# Проверка числовых переменных:

# In[113]:


data_type = 'int'
value_check(table, data_type)


# In[114]:


int_negative_value_update(table)


# Нарушения по числовым переменным также не обнаружены

# Проверим значениях в полях дат:

# In[115]:


data_type = 'date'
value_check(table, data_type)


# In[116]:


date_value_update(table)


# В таблице контрактов некорректных значений в формате даты нет

# In[ ]:





# #### Блок Лапшовой Полины

# ## Кодировка текстовых полей

# In[62]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

def encoding_and_creating_dictionary(table):
    le = preprocessing.LabelEncoder()
    # создаём таблицу для словаря
    cursor.execute("""CREATE TABLE IF NOT EXISTS {0}_dict (column_name VARCHAR(100), category VARCHAR(100), encoding INT)""".format(table))
    # кодируем все текстовые колонки
    cursor.execute("""SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS
    WHERE DATA_TYPE = 'varchar' AND TABLE_SCHEMA = '{0}'
    AND TABLE_NAME = '{1}'""".format(database, table))
    data_type_col = pd.DataFrame(cursor.fetchall())[0].tolist()

    for i in range (0,len(data_type_col)):
        cursor.execute("""SELECT id, {0} FROM {1}""".format(data_type_col[i], table))
        data = pd.DataFrame(cursor.fetchall())
        data.columns = [x[0] for x in cursor.description]
        data = data.dropna(how = 'any').reset_index(drop = True)
        data['encoded'] = le.fit_transform(data[data_type_col[i]])
        cursor.execute("""ALTER TABLE {0} DROP {1}""".format(table, data_type_col[i]))
        cursor.execute("""ALTER TABLE {0} ADD {1} INT""".format(table, data_type_col[i]))
        for j in range (0, len (data)):
            cursor.execute("""UPDATE {0} SET {1} = {2} 
            WHERE {0}.id={3}""".format(table, data_type_col[i], data['encoded'][j], data['id'][j]))
            cnx.commit()
            
            
        # создаём для текущей текстовой колонки таблицу с данными для заполнения словаря
        to_dict = data[[data_type_col[i],'encoded']].drop_duplicates().sort_values(by = ['encoded']).reset_index(drop = True)
        to_dict.columns = ['category','encoding']
        to_dict['column_name'] = data_type_col[i]
        to_dict = to_dict[['column_name', 'category', 'encoding']]

        # заполняем подготовленными данными по текущему текстовому столбцу словарь кодировки
        sql = """INSERT INTO {0}_dict VALUES (%s, %s, %s)""".format(table)
        for k in range(0, to_dict.shape[0]):
            val = (to_dict.iloc[k][0], to_dict.iloc[k][1], int(to_dict.iloc[k][2]))
            cursor.execute(sql, val)

            cnx.commit()


# In[63]:


encoding_and_creating_dictionary('applications')


# In[64]:


encoding_and_creating_dictionary('contracts')


# ## Выгрузка данных в форматы csv

# In[64]:


cnx = mysql.connector.connect(host = 'localhost', database = database, user = user)
cursor = cnx.cursor(buffered=True)

applications_df = pd.read_sql_query("SELECT * FROM applications", cnx)
contracts_df = pd.read_sql_query("SELECT * FROM contracts", cnx)
applications_dict_df = pd.read_sql_query("SELECT * FROM applications_dict", cnx)
contracts_dict_df = pd.read_sql_query("SELECT * FROM contracts_dict", cnx)
cnx.close()


# In[38]:


applications_df.to_csv(my_path + '\\output_1\\applications_df.csv', header=True)
contracts_df.to_csv(my_path + '\\output_1\\contracts_df.csv', header=True)
applications_dict_df.to_csv(my_path + '\\output_1\\applications_dict_df.csv', header=True)
contracts_dict_df.to_csv(my_path + '\\output_1\\contracts_dict_df.csv', header=True)


# In[117]:


cnx.close()


# In[4]:


print("""part_1 выполнена, таблицы анкет, контрактов, словари для их расшифровки и таблица платежей добавлены к базе данных, 
посмотрите на результаты в папке output_1""")


# In[ ]:




