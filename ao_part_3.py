# Блок присвоения рейтинговых баллов (переписанный А. Ощепковым)

# Используемые библиотеки
import pandas as pd
import mysql.connector as connector
import numpy as np
import re
import traceback
pd.options.display.max_rows = 200
pd.options.display.max_columns = 20


# Часть кода, взятой из 1 и 2 частей проекта (для определения названия БД и пользовательских настроек)


def user_input():
    my_path = input('''
        Введите путь папки, в которой будут лежать все ваши файлы
        (пример формата C:\\Users\\User\\Documents\\Python Scripts\\IT\\Untitled Folder):
    ''')
    while my_path  == '':
        print("Ошибка: вы не ввели путь папки")
        my_path = input('''
            Введите путь папки, в которой будут лежать все ваши файлы
            (пример формата C:\\Users\\User\\Documents\\Python Scripts\\IT\\Untitled Folder):
        ''')
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
    return my_path, user, database


# Подключение к БД и вытаскивание таблицы по платежам, которая была получена на 1 и 2 этапе


def connection_to_db_payments(user, database):
    conn = connector.connect(user=user,
                         database=database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM payments;')
    df = pd.DataFrame(cursor.fetchall(),
                      columns=[
        'contract_number', 'payment_date', 'amount_due', 'amount_paid'
    ])
    conn.close()
    return df


# Определение дефолтеров: создается две дамми-переменных: временаня default_suspect - подозреваемый момент
# времени в дефолте (если выплата в этот период меньше, чем было обещано),
# а также default_dummy - фиксация конкретного времени дефолта (не было платежей 90+ дней)


def default_dummy_add(df):
    df['diff'] = df['amount_paid'] - df['amount_due']
    df['default_suspect'] = 0
    df['default_dummy'] = 0
    for j in df['contract_number'].unique():
        df.loc[np.logical_and(df['contract_number'] == j, df['diff'] < 0), 'default_suspect'] = 1
        df.loc[np.logical_and(
            np.logical_and(
                np.logical_and(
                    df['default_suspect'].shift(4) == 0,
                    df['default_suspect'].shift(3) == 1),
                np.logical_and(
                    df['default_suspect'].shift(2) == 1,
                    df['default_suspect'].shift(1) == 1)),
            df['default_suspect'] == 1), 'default_dummy'] = 1
    df = df.drop(df['default_suspect'])
    return df


# Импорт таблицы по качественным показателям (доход, возраст и так далее) из анкет


def applications_import(user, database):
    conn = connector.connect(user=user,
                             database=database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM applications;')
    df = pd.DataFrame(cursor.fetchall(),
                      columns=[
                          'id', 'income', 'children', 'family', 'application_date', 'age', 'income_type',
                          'housing', 'house_ownership', 'marital_status', 'gender', 'employed_by', 'education',
                          'position', 'car_owner'
                      ])
    conn.close()
    return df


# Функция импорта табоицы по контрактам, чтобы связать дефолты с ID-шниками дефолтеров


def contracts_import(user, database):
    conn = connector.connect(user=user,
                             database=database)
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM contracts;')
    df = pd.DataFrame(cursor.fetchall(),
                      columns=[
                          'id', 'amount', 'term_month', 'contract_number',
                          'annuity', 'contract_date', 'type'
                      ])
    conn.close()
    return df


# Функция для объединения всех полученных таблиц по ключам contract_number & id


def data_manipulation(df, dfContr, dfAppl):
    defaults = pd.DataFrame(df.loc[df['default_dummy'] == 1, 'contract_number'],
                            columns=['contract_number'])    # таблица по номеру контракта - факту дефолта
    defaults['dummy_default'] = 1    # то же
    contract_numbers_defaults = pd.DataFrame(df['contract_number'].unique(), columns=['contract_number'])    # то же
    contract_numbers_defaults['dummy_default'] = 'NaN'    # то же
    for j in defaults.contract_number:    # то же
        contract_numbers_defaults.loc[contract_numbers_defaults['contract_number'] == j] = 'NaN'    # то же
    contract_numbers_defaults = contract_numbers_defaults.loc[contract_numbers_defaults['contract_number'] != 'NaN']
    defaults = pd.concat([defaults, contract_numbers_defaults])    # то же
    defaults.loc[defaults['dummy_default'] == 'NaN', 'dummy_default'] = 0    # то же
    defaults = defaults.reset_index(drop=True)    # то же

    dfContr = dfContr.set_index('contract_number')    # объединение всех таблиц в одну по ключам contract_number & id
    defaults = defaults.set_index('contract_number')    # то же
    df = dfContr.join(defaults, how='inner')    # то же
    df = df.reset_index()    # то же
    df = df.set_index('id')    # то же
    dfAppl = dfAppl.set_index('id')    # то же
    df = df.join(dfAppl, how='inner')    # то же
    df = df.reset_index()    # то же
    return df


# Функции, взятые из блоков 1 и 2 для расчета показателей WOE, IV


def char_bin(Y, X):
    df1 = pd.DataFrame({"X": X, "Y": Y})
    justmiss = df1[['X', 'Y']][df1.X.isnull()]
    notmiss = df1[['X', 'Y']][df1.X.notnull()]
    df2 = notmiss.groupby('X', as_index=True)

    d3 = pd.DataFrame({}, index=[])
    d3["COUNT"] = df2.count().Y
    d3["MIN_VALUE"] = df2.sum().Y.index
    d3["MAX_VALUE"] = d3["MIN_VALUE"]
    d3["EVENT"] = df2.sum().Y
    d3["NONEVENT"] = df2.count().Y - df2.sum().Y

    if len(justmiss.index) > 0:
        d4 = pd.DataFrame({'MIN_VALUE':np.nan}, index=[0])
        d4["MAX_VALUE"] = np.nan
        d4["COUNT"] = justmiss.count().Y
        d4["EVENT"] = justmiss.sum().Y
        d4["NONEVENT"] = justmiss.count().Y - justmiss.sum().Y
        d3 = d3.append(d4, ignore_index=True, sort=False)

    d3["EVENT_RATE"] = d3.EVENT/d3.COUNT
    d3["NON_EVENT_RATE"] = d3.NONEVENT/d3.COUNT
    d3["DIST_EVENT"] = (d3.EVENT + 0.5)/d3.sum().EVENT
    d3["DIST_NON_EVENT"] = (d3.NONEVENT + 0.5)/d3.sum().NONEVENT
    d3["WOE"] = np.log((d3.DIST_EVENT) / (d3.DIST_NON_EVENT))
    d3["IV"] = (d3.DIST_EVENT-d3.DIST_NON_EVENT)*np.log(d3.DIST_EVENT/d3.DIST_NON_EVENT)
    d3["VAR_NAME"] = "VAR"
    d3 = d3[['VAR_NAME', 'MIN_VALUE', 'MAX_VALUE', 'COUNT', 'EVENT', 'EVENT_RATE', 'NONEVENT', 'NON_EVENT_RATE',
             'DIST_EVENT', 'DIST_NON_EVENT', 'WOE', 'IV']]
    d3 = d3.replace([np.inf, -np.inf], 0)
    d3.IV = d3.IV.sum()
    d3 = d3.reset_index(drop=True)
    return d3


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
                iv_df = iv_df.append(conv, ignore_index=True, sort=False)

    iv = pd.DataFrame({'IV': iv_df.groupby('VAR_NAME').IV.max()})
    iv = iv.reset_index()
    return iv_df, iv


# Функция для составления таблицы WOE, IV по всем переменным


def woe_full(df):
    df1 = df.drop(
        [  # удаляем все не интересующие нас поля (скоринг будем проводить по характеристикам заемщиков из анкет)
            'contract_number', 'amount', 'term_month', 'annuity', 'contract_date', 'type',  # то же
            'application_date'], axis=1)  # то же
    df1['income'] = pd.cut(df1['income'], 5, labels=[1, 2, 3, 4, 5])  # непрерывные величины -> дискретные
    df1['age'] = pd.cut(df1['age'], 5, labels=[1, 2, 3, 4, 5])  # то же
    df1['employed_by'] = pd.cut(df1['employed_by'], 5, labels=[1, 2, 3, 4, 5])  # то же
    df2 = df1.drop(['id', 'dummy_default'], axis=1)  # проведение оценки WOE, IV
    df_woe = woe_iv_categorical(df2, df1['dummy_default'])[0]  # проведение оценки WOE, IV
    return df1, df_woe


# Функция для составления скоринга по переменным в соответствии с возрастанием WOE (от 1 до 12 -> в шкалу до 100%)


def scoring(df_woe):
    df_woe2 = df_woe
    df_woe2 = df_woe2.sort_values(by=['VAR_NAME', 'WOE']).reset_index(drop=True)
    df_woe2['SCORE'] = 0
    df_woe2.loc[df_woe2['VAR_NAME'] == 'age', 'SCORE'] = [
        1 * 100 / 12, 5 * 100 / 12, 8 * 100 / 12, 10 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'car_owner', 'SCORE'] = [
        3 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'children', 'SCORE'] = [
        1 * 100 / 12, 5 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'education', 'SCORE'] = [
        1 * 100 / 12, 5 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'employed_by', 'SCORE'] = [
        1 * 100 / 12, 5 * 100 / 12, 8 * 100 / 12, 10 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'family', 'SCORE'] = [
        1 * 100 / 12, 4 * 100 / 12, 7 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'gender', 'SCORE'] = [
        1 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'house_ownership', 'SCORE'] = [
        1 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'housing', 'SCORE'] = [
        1 * 100 / 12, 4 * 100 / 12, 7 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'income', 'SCORE'] = [
        1 * 100 / 12, 5 * 100 / 12, 8 * 100 / 12, 10 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'income_type', 'SCORE'] = [
        1 * 100 / 12, 4 * 100 / 12, 7 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'marital_status', 'SCORE'] = [
        1 * 100 / 12, 4 * 100 / 12, 7 * 100 / 12, 9 * 100 / 12, 12 * 100 / 12]
    df_woe2.loc[df_woe2['VAR_NAME'] == 'position', 'SCORE'] = [
        1 * 100 / 12, 2 * 100 / 12, 3 * 100 / 12, 4 * 100 / 12, 5 * 100 / 12,
        6 * 100 / 12, 7 * 100 / 12, 8 * 100 / 12, 9 * 100 / 12, 10 * 100 / 12,
        11 * 100 / 12, 12 * 100 / 12]
    return df_woe2


# Функция для составления итоговой таблицы со 100-бальной шкалой по всем переменным


def score_final(df1, df_woe2):
    output = pd.DataFrame(data=None, columns=[
        'id', 'age', 'car_owner', 'children', 'education', 'employed_by',
        'family', 'gender', 'house_ownership', 'housing', 'income', 'income_type',
        'marital_status', 'position'])
    output['id'] = df1['id']
    l = output.columns.values.tolist()[1:]
    for j in output.id:
        for k in l:
            x = df_woe2.loc[df_woe2['VAR_NAME'] == k, ['MIN_VALUE', 'SCORE']]
            y = df1.loc[df1['id'] == j, k].values.tolist()
            if np.isnan(y):
                z = x.loc[x['MIN_VALUE'].isnull(), 'SCORE'].values.tolist()
            else:
                z = x.loc[x['MIN_VALUE'] == y, 'SCORE'].values.tolist()
            output.loc[output['id'] == j, k] = z
    return output

# Вызов всех функций в главной части программы
my_path, user, database = user_input()    # пользовательские настройки
df = connection_to_db_payments(user, database)    # импорт таблицы по платежам из БД
df = default_dummy_add(df)    # добавленное поле с dummy по дефолтам
dfAppl = applications_import(user, database)    # Импорт таблицы по анкетам
dfContr = contracts_import(user, database)    # Импорт таблицы по контрактам
df = data_manipulation(df, dfContr, dfAppl)   # Объединение таблиц по анкетам, контрактам и дефолтам (частью платежей)
df1, df_woe = woe_full(df)   # Составление таблицы WOE по всем переменным
df_woe2 = scoring(df_woe)    # Получение бальной шкалы по всем переменным в соответствии с их группами
output = score_final(df1, df_woe2)    # Составление итогой таблицы с score-значениями по всем переменным
defaults = df[['id', 'dummy_default']]    # Подготовка таблицы для выгрузки по дефолтам
defaults = defaults.rename(index=str, columns={"dummy_default": "default_"})    # то же
df_woe.to_csv(my_path + '/output_3/woe_iv_table.csv')    # выгрузка 1
output.to_csv(my_path + '/output_3/scores_for_vars.csv')    # выгрузка 2
defaults.to_csv(my_path + '/output_3/defaults_storage.csv')    # выгрузка 3
print('\nБлок выполнен. Результаты экспортированы в output_3\n')
