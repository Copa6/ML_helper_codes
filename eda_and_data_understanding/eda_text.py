import pandas as pd
import numpy as np
import seaborn as sns
import regex as re 
import spacy
import nltk
from nltk.corpus import stopwords

# ### If nltk data is not downloaded ###
# nltk.download('stopwords')

def generate_custom_features(data, column=None):
	text_data = data[column] if column else data.iloc[:,0]
	
	return_data = pd.DataFrame()

	return_data[f"{column}_count_chars"] = [len(txt) for txt in text_data]
	return_data[f"{column}_count_words"] = [len(txt.split()) for txt in text_data]
	return_data[f"{column}_count_uppercases"] = text_data.apply(lambda t: sum([w[0].isupper() for w in t.split()]))

	puncts_regexp = re.compile(r'[\p{P}]+')
	return_data[f"{column}_count_punct"] = text_data.apply(lambda t: len(re.findall(puncts_regexp, t)))

	numbers_regexp = re.compile(r'[0-9]')
	return_data[f"{column}_count_numbers"] = text_data.apply(lambda t: len(re.findall(numbers_regexp, t)))

	return return_data