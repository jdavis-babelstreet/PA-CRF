"""
This script contains utility functions to use with computing data set and model stats as well as any
visualization helpers
"""
import time
import numpy as np
from sklearn.manifold import TSNE
import json
import pandas as pd


def generate_dataset_output(data_path=None, tokenizer=None):
    raw_data = json.load(open(data_path, 'r'))
    classes = raw_data.keys()
    raw_tokens = []
    raw_text = []
    tsne_embeddings = {}
    # Setting perplexity because it must be less than the number of samples.
    # The default value of 30.0 was > num samples of a class
    tsne = TSNE(n_components=3, verbose=0, perplexity=20.0)
    for i, class_name in enumerate(classes):
        sentences = []
        for j in range(len(raw_data[class_name])):
            tokens = raw_data[class_name][j]['tokens']
            sent = ' '.join(t for t in tokens)
            raw_tokens.append(tokens)
            raw_text.append(sent)
            sentences.append(sent)
        # Need to pad the tokens in order to convert them to a NDArray for use with TSNE
        enc = tokenizer(sentences, padding='longest')['input_ids']
        enc = np.asarray(enc)
        tsne_enc = tsne.fit_transform(enc)
        tsne_embeddings[class_name] = tsne_enc
    start_time = time.perf_counter()
    encodings = tokenizer(raw_text)
    end_time = time.perf_counter()
    encoded_tokens = encodings['input_ids']
    tokenization_time = (end_time - start_time)
    avg_raw_token_len = np.mean([len(s) for s in raw_tokens])
    avg_enc_token_len = np.mean([len(t) for t in encoded_tokens])
    avg_word_len = np.mean([len(l) for rt in raw_tokens for l in rt])
    num_samples = len(raw_text)
    stats = {
        'avg_raw_token_len': avg_raw_token_len,
        'avg_enc_token_len': avg_enc_token_len,
        'avg_word_len': avg_word_len,
        'num_samples': num_samples,
        'tokenization_time': tokenization_time
    }
    return stats, tsne_embeddings


def dump_tokenizations_to_json(output_path=None, tokenizations=None):
    with open(output_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in tokenizations.items()}, f)


def load_tokenizations_from_json(input_path=None):
    # See: https://stackoverflow.com/a/65339713
    enc_tokens = json.load(open(input_path, 'r'))
    enc_tokens = {k: np.array(v) for k, v in enc_tokens.items()}
    return enc_tokens


def create_df_from_tokenizations(tokenization=None):
    res_list = []
    class_names = tokenization.keys()
    for i, class_name in enumerate(class_names):
        for j in range(len(tokenization[class_name])):
            x = tokenization[class_name][j][0]
            y = tokenization[class_name][j][1]
            z = tokenization[class_name][j][2]
            res_list.append({
                'class_name': class_name,
                'x': x,
                'y': y,
                'z': z
            })
    df = pd.DataFrame.from_dict(res_list)
    return df


def generate_embeddings(data_path=None, tokenizer=None, encoder=None):
    """
    Process the raw datasets again and use the tokenized inputs as input to the Bert model.
    Grab the CLS token from the Bert model and perform TSNE on it to generate the 3 principal components
    Return the TSNE embeddings for use with Plotly visualizations.
    :param data_path:
    :param tokenizer:
    :param encoder:
    :return:
    """
    encoder.eval()  # Don't need this trying to compute gradients
    raw_data = json.load(open(data_path, 'r'))
    classes = raw_data.keys()
    tsne_embeddings = {}
    cls_embeddings = {}
    tsne = TSNE(n_components=3, verbose=0, perplexity=20.0)  # See above about perplexity
    for i, class_name in enumerate(classes):
        sentences = []
        for j in range(len(raw_data[class_name])):
            tokens = raw_data[class_name][j]['tokens']
            sent = ' '.join(t for t in tokens)
            sentences.append(sent)
        encodings = tokenizer(sentences, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
        encodings = encodings['input_ids']
        batch_embeddings = []
        for x in batch(encodings):
            outputs = encoder(x)
            cls_emb = outputs['pooler_output'].detach().numpy()
            batch_embeddings.append(cls_emb)
        class_embeddings = np.concatenate(batch_embeddings)
        tsne_emb = tsne.fit_transform(class_embeddings)
        cls_embeddings[class_name] = class_embeddings
        tsne_embeddings[class_name] = tsne_emb
    return cls_embeddings, tsne_embeddings


def dump_embeddings_to_json(output_path=None, embeddings=None):
    with open(output_path, 'w') as f:
        json.dump({k: v.tolist() for k, v in embeddings.items()}, f)


def load_embeddings_from_json(input_path=None):
    embeddings = json.load(open(input_path, 'r'))
    embeddings = {k: np.array(v) for k, v in embeddings.items()}
    return embeddings


def batch(data, batch_size=16):
    # See: https://stackoverflow.com/a/8290508
    l = len(data)
    for idx in range(0, l, batch_size):
        yield data[idx:min(idx + batch_size, l)]


def create_df_from_embeddings(embeddings=None):
    res_list = []
    class_names = embeddings.keys()
    for i, class_name in enumerate(class_names):
        for j in range(len(embeddings[class_name])):
            x = embeddings[class_name][j][0]
            y = embeddings[class_name][j][1]
            z = embeddings[class_name][j][2]
            res_list.append({
                'class_name': class_name,
                'x': x,
                'y': y,
                'z': z
            })
    df = pd.DataFrame.from_dict(res_list)
    return df

