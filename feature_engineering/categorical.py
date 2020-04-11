def label_encode_data(data, columns, save_model=True, model_name="label_encoder"):
	from sklearn.preprocessing import LabelEncoder
	from joblib import dump

	return_data = pd.DataFrame()
	models = {}
	for column in columns:
		encoder = LabelEncoder()
		encoder.fit(data[column])
		return_data[f"transformed_{column}"] = encoder.transform(data[column])
		models[column] = encoder
	
	if save_model:
		dump(models, f"{model_name}.model")

	return return_data

def label_encode_from_model(data, columns, model_file=None, model=None):
	from joblib import load
	if model_file:
		encoders = load(model_file)
	else:
		encoders = model

	return_data = pd.DataFrame()
	for column in columns:
		encoder = encoders[column]
		return_data[f"transformed_{column}"] = encoder.transform(data[column])

	return return_data