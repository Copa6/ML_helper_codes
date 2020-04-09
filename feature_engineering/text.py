def preprocess_text(data, column=None, lemmatize=True, remove_punctuations_number_url=True):
   	nlp = spacy.load('en', disable=['parser', 'ner'])
	stop_words = set(stopwords.words('english'))

	return_data = pd.DataFrame()
	text_data = data[column] if column else data.iloc[:,0]
	new_col = []
	for i, row_data in enumerate(text_data):
		row_data = str(row_data).strip().lower()  # Lower case the data
		if len(row_data) != 0:
			if lemmatize:
				parsed_data = nlp(row_data)
				data_lemmatized = [w.lemma_ if '-PRON-' not in w.lemma_ else w.text for w in parsed_data if not w.is_stop]

				row_data_lemmatized = ' '.join(data_lemmatized)
			else:
				row_data_lemmatized = row_data

			if remove_punctuations_number_url:
				# Remove url, numbers, punct
				row_data_punct_removed = re.sub(
					r'((http|ftp|https|mailto)*://[A-Za-z0-9._?=+&%-]*(/[A-Za-z0-9._?=+&%-]*)*)|(www.*?\s+)|(.*?@.*?\s+)|[0-9]|[^\w\s]',
					'',
					str(row_data_lemmatized))
			else:
				row_data_punct_removed = row_data	
		else:
			row_data_punct_removed = row_data
		
		new_col.append(row_data_punct_removed)
	return_data[f"preprocessed_{col_name}"] = new_col

	return return_data