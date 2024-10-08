import pandas as pd
from main import EXCEL_FOLDER

train = pd.read_csv(EXCEL_FOLDER + '\\SemEval_Laptop_train.csv')
test = pd.read_csv(EXCEL_FOLDER + '\\SemEval_Laptop_test.csv')

train_pivot_table = train.pivot_table(index='aspect', columns='sentiment', aggfunc='size', fill_value=0)
test_pivot_table = test.pivot_table(index='aspect', columns='sentiment', aggfunc='size', fill_value=0)
ordered_columns = [1, 0, 2]

train_pivot_table = train_pivot_table[ordered_columns]
test_pivot_table = test_pivot_table[ordered_columns]

train_pivot_table['sum'] = train_pivot_table.sum(axis=1)
test_pivot_table['sum'] = test_pivot_table.sum(axis=1)

train_pivot_table.loc['Total'] = train_pivot_table.sum()
test_pivot_table.loc['Total'] = test_pivot_table.sum()

train_pivot_table.to_csv(EXCEL_FOLDER + '\\pivottable_laptop_train.csv')
test_pivot_table.to_csv(EXCEL_FOLDER + '\\pivottable_laptop_test.csv')

a = 1

