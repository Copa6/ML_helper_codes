def scale_data(data, columns, save_model=True, model_name="standard_scaler"):
	import pandas as pd
	from sklearn.preprocessing import MinMaxScaler
	from joblib import dump

	scaler = MinMaxScaler()
	scaler.fit(data[columns])
	transformed_columns = scaler.transform(data[columns])
	if save_model:
		dump(scaler, f"{model_name}.model")

	return pd.DataFrame(transformed_columns)


def scale_data_from_model(data, columns, model_file=None, model=None):
	import pandas as pd
	from joblib import load
	if model_file:
		scaler = load(model_file)
	else:
		scaler = model
	transformed_columns = scaler.transform(data[columns])

	return pd.DataFrame(transformed_columns)


def scale_values_by_group(data, value_colum, scaling_data):
	pass