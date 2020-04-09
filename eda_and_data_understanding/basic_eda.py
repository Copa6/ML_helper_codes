import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from datetime import datetime



def data_summary(data, max_uniques=10):
	nrows, ncols = data.shape
	print(f"{'*'*10}Print summaries for train data{'*'*10}")
	print(f"Data has {nrows} rows and {ncols} columns")
	
 	print(f"Finding number of unique values per column")
	for col in train.columns:
		num_uniques = data[col].nunique()
		if num_uniques <= max_uniques:
			print(f"{col} - {num_uniques} {data[col].unique()}")
		else:
			print(f"{col} - {num_uniques}")

	print("Numeric data distributions")
	print(train.describe())

	print("Data Information")
	print(data.info())

	print(data.head())

def plot_column_counts_by_category(data, columns=None, missing_fill=-1, max_uniques=10):
	sns.set_style("white")
	plot_columns = columns if columns else data.columns
	for column in plot_columns:
		if data[column].nunique() <= max_uniques:
			plt.figure()
			sns.countplot(data[column].fillna(missing_fill))


def categorical_target_count_plot(data, target_column, columns=None, missing_fill=-1, max_uniques=10):
	sns.set_style("white")
	updated_data = data.copy().fillna(missing_fill)
	plot_columns = columns if columns else data.columns
	for column in columns:
		if data[column].nunique() <= max_uniques:
			plt.figure()
			sns.catplot(x=column, hue=target_column, kind="count", data=updated_data);


if __name__=="__main__":
	data = pd.read_csv("data_csv_path.csv", sep=",")
	target_var = "target"

	data_summary(data)
	plot_column_counts_by_category(data)

	categorical_target_count_plot(data, target_column=var)



