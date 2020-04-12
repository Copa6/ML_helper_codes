def generate_custom_features(data, column=None):
	import pandas as pd
	import regex as re
	text_data = data[column] if column else data.iloc[:,0]
	
	return_data = pd.DataFrame()

	return_data[f"{column}_count_chars"] = [len(txt) for txt in text_data]
	return_data[f"{column}_count_words"] = [len(txt.split()) for txt in text_data]
	return_data[f"{column}_count_uppercases"] = text_data.apply(lambda t: sum([w[0].isupper() for w in t.split()]))

	puncts_regexp = re.compile(r'[\p{P}]+')
	return_data[f"{column}_count_punct"] = text_data.apply(lambda t: len(re.findall(puncts_regexp, t)))

	numbers_regexp = re.compile(r'[0-9]')
	return_data[f"{column}_count_numbers"] = text_data.apply(lambda t: len(re.findall(numbers_regexp, t)))

	links_regexp = re.compile(r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*')
	return_data[f"{column}_count_links"] = text_data.apply(lambda t: len(re.findall(links_regexp, t)))

	return return_data
