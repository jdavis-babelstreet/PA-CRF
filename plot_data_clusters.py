import plotly.express as px
import pandas as pd
import os

tokenization_dir = './checkpoint/results/tokenization/'
train_tok_path = f'{tokenization_dir}training.csv'
val_tok_path = f'{tokenization_dir}validation.csv'
test_tok_path = f'{tokenization_dir}testing.csv'
train_tok_png = f'{tokenization_dir}training.png'
val_tok_png = f'{tokenization_dir}validation.png'
test_tok_png = f'{tokenization_dir}testing.png'

embeddings_dir = './checkpoint/results/embeddings/'
train_emb_path = f'{embeddings_dir}training.csv'
val_emb_path = f'{embeddings_dir}validation.csv'
test_emb_path = f'{embeddings_dir}testing.csv'
train_emb_png = f'{embeddings_dir}training.png'
val_emb_png = f'{embeddings_dir}validation.png'
test_emb_png = f'{embeddings_dir}testing.png'


def main():
    """
       Can only create one graph at a time because plotly opens the graphs in a browser window.
       I'll check if the png files exist and then create the graphs accordingly.
       These plots are interactive in the browser.
       """

    if not os.path.exists(train_tok_png):
        train_tok_df = pd.read_csv(train_tok_path, encoding='utf-8', engine='python')
        fig = px.scatter_3d(train_tok_df, x='x', y='y', z='z', color='class_name', title='Training Tokenization')
        fig.show()
        return
    if not os.path.exists(val_tok_png):
        val_tok_df = pd.read_csv(val_tok_path, encoding='utf-8', engine='python')
        fig = px.scatter_3d(val_tok_df, x='x', y='y', z='z', color='class_name', title='Validation Tokenization')
        fig.show()
        return
    if not os.path.exists(test_tok_png):
        test_tok_df = pd.read_csv(test_tok_path, encoding='utf-8', engine='python')
        fig = px.scatter_3d(test_tok_df, x='x', y='y', z='z', color='class_name', title='Testing Tokenization')
        fig.show()
        return

    if not os.path.exists(train_emb_png):
        train_emb_df = pd.read_csv(train_emb_path, encoding='utf-8', engine='python')
        fig = px.scatter_3d(train_emb_df, x='x', y='y', z='z', color='class_name', title='Training Embeddings')
        fig.show()
        return
    if not os.path.exists(val_emb_png):
        val_emb_df = pd.read_csv(val_emb_path, encoding='utf-8', engine='python')
        fig = px.scatter_3d(val_emb_df, x='x', y='y', z='z', color='class_name', title='Validation Embeddings')
        fig.show()
        return
    if not os.path.exists(test_emb_png):
        test_emb_df = pd.read_csv(test_emb_path, encoding='utf-8', engine='python')
        fig = px.scatter_3d(test_emb_df, x='x', y='y', z='z', color='class_name', title='Testing Embeddings')
        fig.show()
        return


if __name__ == '__main__':
    main()


