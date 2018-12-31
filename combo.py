#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os
file = 0
while str(file) != 'выход':
    file = input("Введите название части проекта, которую вы хотите запустить (из part_1, part_2, part_3, part_4), либо введите слово выход: ")
    while file  == '':
        print("Ошибка: вы не ввели название файла для запуска")
        file = input("Введите название части проекта, которую вы хотите запустить (из part_1, part_2, part_3, part_4), либо введите слово выход: ")
    if file != 'выход':
        print("Вы ввели '" + str(file)+"'")
    os.system("python "+file+".py")
if file == 'выход':
    print("Вы ввели '" + str(file)+"', до свидания!")


# In[ ]:




