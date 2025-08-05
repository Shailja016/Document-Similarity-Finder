#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install streamlit')


# In[1]:


where streamlit


# In[ ]:


get_ipython().system('pip install ijson')
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


# In[ ]:


# Set Streamlit page config
st.set_page_config(page_title="Document Similarity Finder", layout="wide")


# In[ ]:


# Load SBERT model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')


# In[ ]:


model = load_model()


# In[ ]:




