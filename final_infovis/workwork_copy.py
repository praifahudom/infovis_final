import streamlit as st
import pandas as pd
import networkx as nx
from wordcloud import WordCloud
from pyvis.network import Network
import matplotlib.pyplot as plt
from collections import Counter

# Load the CSV file without using sentence ID
data = pd.read_csv('/Users/littleflower/anaconda3/final_infovis/thai_addresses_with_predictions.csv')

# Split data by legend (LOC, ADDR, POST, O)
loc_words = data[data['Predicted Tag'] == 'LOC']['Token']
addr_words = data[data['Predicted Tag'] == 'ADDR']['Token']
post_words = data[data['Predicted Tag'] == 'POST']['Token']
o_words = data[data['Predicted Tag'] == 'O']['Token']

# Create WordClouds for each legend
def plot_wordcloud(words, title):
    if len(words) > 0:
        word_freq = Counter(words)
        wordcloud = WordCloud(width=800, height=400, background_color='white', font_path='Noto_Sans_Thai copy/NotoSansThai-VariableFont_wdth,wght.ttf').generate_from_frequencies(word_freq)
        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        st.pyplot(plt)
    else:
        st.write(f"No words available for {title} category.")

# Streamlit UI
st.title('Word Cloud and Sentence Relationship Visualization')

# Display Word Clouds
st.header('Word Clouds by Legend')
plot_wordcloud(loc_words, 'LOC')
plot_wordcloud(addr_words, 'ADDR')
plot_wordcloud(post_words, 'POST')
plot_wordcloud(o_words, 'O')

# Create graph visualization for relationships between words
st.header('Word Relationship Graph')
G = nx.Graph()

# Add nodes and edges for words in the same sequence without using ID
words = data['Token'].tolist()
for i, word in enumerate(words):
    G.add_node(word)
    if i > 0:
        G.add_edge(words[i - 1], word)

# Visualize using Pyvis
net = Network(notebook=True, height='500px', width='100%')
net.from_nx(G)
net.show('word_relationship_graph.html')

# Display the graph in Streamlit
st.components.v1.html(open('word_relationship_graph.html', 'r').read(), height=500, scrolling=True)