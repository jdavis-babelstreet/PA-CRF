"""
Added this script in to use to compute timings for the trained models during inference.
"""
import os
import time
import json
import numpy as np
import pandas as pd

from transformers import BertTokenizerFast, BertModel
from result_utils import generate_dataset_output, dump_tokenizations_to_json, load_tokenizations_from_json
from result_utils import create_df_from_tokenizations, generate_embeddings, dump_embeddings_to_json
from result_utils import load_embeddings_from_json, create_df_from_embeddings

# Set the huggingface download directory
HF_CACHE_DIR = './checkpoint/hf_models/'
os.environ['TRANSFORMERS_CACHE'] = HF_CACHE_DIR
if not os.path.exists(HF_CACHE_DIR):
    os.makedirs(HF_CACHE_DIR)


def main():
    # Set up the file paths for the data sets
    train_path = './data/FewEvent/meta_train_dataset.json'
    val_path = './data/FewEvent/meta_dev_dataset.json'
    test_path = './data/FewEvent/meta_test_dataset.json'
    encoder_path = 'bert-base-uncased'
    results_file = './checkpoint/results/dataset_stats.csv'
    tokenization_dir = './checkpoint/results/tokenization/'
    train_token_path = tokenization_dir + 'training.json'
    val_token_path = tokenization_dir + 'validation.json'
    test_token_path = tokenization_dir + 'testing.json'
    if not os.path.exists(tokenization_dir):
        os.makedirs(tokenization_dir)
    embeddings_dir = './checkpoint/results/embeddings/'
    if not os.path.exists(embeddings_dir):
        os.makedirs(embeddings_dir)

    # Process the datasets if they haven't already been processed. If the stats file exists, they have
    if not os.path.exists(results_file):
        stats = []
        tokenizer = BertTokenizerFast.from_pretrained(encoder_path, cache_dir=HF_CACHE_DIR)

        # Training
        data_stats, train_tsne_tokenizations = generate_dataset_output(train_path, tokenizer)
        data_stats['dataset'] = 'training'
        stats.append(data_stats)

        # Validation
        data_stats, val_tsne_tokenizations = generate_dataset_output(val_path, tokenizer)
        data_stats['dataset'] = 'validation'
        stats.append(data_stats)

        # Testing
        data_stats, test_tsne_tokenizations = generate_dataset_output(test_path, tokenizer)
        data_stats['dataset'] = 'testing'
        stats.append(data_stats)

        df = pd.DataFrame.from_dict(stats)
        df.to_csv(results_file, encoding='utf-8', header=True, index=False)

        # Write the tsne tokenizations to file
        dump_tokenizations_to_json(train_token_path, train_tsne_tokenizations)
        dump_tokenizations_to_json(val_token_path, val_tsne_tokenizations)
        dump_tokenizations_to_json(test_token_path, test_tsne_tokenizations)

    # Compute dataframes from each of the tokenizations dictionaries to use for visualization if they do not exist
    # If any of the CSV files exist, they have been computed
    train_df_path = tokenization_dir + 'training.csv'
    val_df_path = tokenization_dir + 'validation.csv'
    test_df_path = tokenization_dir + 'testing.csv'
    if not os.path.exists(train_df_path):
        train_tokenizations = load_tokenizations_from_json(train_token_path)
        train_df = create_df_from_tokenizations(train_tokenizations)
        train_df.to_csv(train_df_path, header=True, index=False, encoding='utf-8')
    if not os.path.exists(val_df_path):
        val_tokenizations = load_tokenizations_from_json(val_token_path)
        val_df = create_df_from_tokenizations(val_tokenizations)
        val_df.to_csv(val_df_path, header=True, index=False, encoding='utf-8')
    if not os.path.exists(test_df_path):
        test_tokenizations = load_tokenizations_from_json(test_token_path)
        test_df = create_df_from_tokenizations(test_tokenizations)
        test_df.to_csv(test_df_path, header=True, index=False, encoding='utf-8')

    # Compute embeddings using BERT and perform TSNE to reduce their dimensions down to 3
    # This will be used for visualization of the linear separability of the data sets
    train_embeddings_path = f'{embeddings_dir}training.json'
    val_embeddings_path = f'{embeddings_dir}validation.json'
    test_embeddings_path = f'{embeddings_dir}testing.json'
    train_tsne_path = f'{embeddings_dir}train_tsne.json'
    val_tsne_path = f'{embeddings_dir}val_tsne.json'
    test_tsne_path = f'{embeddings_dir}test_tsne.json'

    train_df_path = f'{embeddings_dir}training.csv'
    val_df_path = f'{embeddings_dir}validation.csv'
    test_df_path = f'{embeddings_dir}testing.csv'

    tokenizer = BertTokenizerFast.from_pretrained(encoder_path, cache_dir=HF_CACHE_DIR)
    encoder = BertModel.from_pretrained(encoder_path, cache_dir=HF_CACHE_DIR)
    # If the json file exists then we don't need to recompute
    if not os.path.exists(train_embeddings_path):
        # Call the generate_embeddings function here.
        train_embeddings, train_tsne_embeddings = generate_embeddings(train_path, tokenizer, encoder)
        dump_embeddings_to_json(train_embeddings_path, train_embeddings)
        dump_embeddings_to_json(train_tsne_path, train_tsne_embeddings)
    elif not os.path.exists(train_df_path):
        # Load the embeddings and generate a data frame
        train_tsne_embeddings = load_embeddings_from_json(train_tsne_path)
        train_tsne_df = create_df_from_embeddings(train_tsne_embeddings)
        train_tsne_df.to_csv(train_df_path, index=False, header=True, encoding='utf-8')
    if not os.path.exists(val_embeddings_path):
        val_embeddings, val_tsne_embeddings = generate_embeddings(val_path, tokenizer, encoder)
        dump_embeddings_to_json(val_embeddings_path, val_embeddings)
        dump_embeddings_to_json(val_tsne_path, val_tsne_embeddings)
    elif not os.path.exists(val_df_path):
        # Load the embeddings and generate a data frame
        # val_embeddings = load_embeddings_from_json(val_embeddings_path)
        val_tsne_embeddings = load_embeddings_from_json(val_tsne_path)
        val_tsne_df = create_df_from_embeddings(val_tsne_embeddings)
        val_tsne_df.to_csv(val_df_path, index=False, header=True, encoding='utf-8')
    if not os.path.exists(test_embeddings_path):
        test_embeddings, test_tsne_embeddings = generate_embeddings(test_path, tokenizer, encoder)
        dump_embeddings_to_json(test_embeddings_path, test_embeddings)
        dump_embeddings_to_json(test_tsne_path, test_tsne_embeddings)
    elif not os.path.exists(test_df_path):
        # Load the embeddings and generate a data frame
        # test_embeddings = load_embeddings_from_json(test_embeddings_path)
        test_tsne_embeddings = load_embeddings_from_json(test_tsne_path)
        test_tsne_df = create_df_from_embeddings(test_tsne_embeddings)
        test_tsne_df.to_csv(test_df_path, index=False, header=True, encoding='utf-8')


if __name__ == '__main__':
    main()
