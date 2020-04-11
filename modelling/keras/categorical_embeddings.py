def create_categorical_embeddings_model():
  inputs = []
  outputs = []
  for c in catcols:
      num_unique_values = int(data[c].nunique())
      embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
      inp = layers.Input(shape=(1,))
      out = layers.Embedding(num_unique_values + 1, embed_dim, name=c)(inp)
      out = layers.SpatialDropout1D(0.3)(out)
      out = layers.Reshape(target_shape=(embed_dim, ))(out)
      inputs.append(inp)
      outputs.append(out)
