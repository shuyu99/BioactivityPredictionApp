import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image
import base64
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.ML.Descriptors import MoleculeDescriptors

# Molecular descriptor calculator
def desc_calc(data):
    smiles = data[0]
    Morgan_fpts = []
    for i in smiles:
        mol = Chem.MolFromSmiles(i)
        fpts = AllChem.GetMorganFingerprintAsBitVect(mol,2,2048)
        mfpts = np.array(fpts)
        Morgan_fpts.append(mfpts)
    Morgan_fpts = np.array(Morgan_fpts)
    Morgan_fingerprints = pd.DataFrame(Morgan_fpts, columns=['Col_{}'.format(i) for i in range(Morgan_fpts.shape[1])])
    Morgan_fingerprints.to_csv('descriptor_output.csv', index=False)

# File download
def filedownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # strings <-> bytes conversions
    href = f'<a href="data:file/csv;base64,{b64}" download="prediction.csv">Download Predictions</a>'
    return href

# Model building
def build_model(input_data):
    # Reads in saved regression model
    load_model = pickle.load(open('beta_lactamase_model.pkl', 'rb'))
    # Apply model to make predictions
    prediction = load_model.predict(input_data)
    st.header('**Prediction output**')
    prediction_output = pd.Series(prediction, name='pIC50')
    molecule_name = pd.Series(load_data[1], name='molecule_name')
    df = pd.concat([molecule_name, prediction_output], axis=1)
    st.write(df)
    st.markdown(filedownload(df), unsafe_allow_html=True)

# Logo image
image = Image.open('logo.png')

st.image(image, use_column_width=True)

# Page title
st.markdown("""

This app allows you to predict the bioactivity value of pIC50 towards inhibiting the `Beta-lactamase`.

Beta-lactamase can lyse beta-lactam rings that are however essential for the activity for a variety of antibiotic classes.

- This learning model's data comes from the beta-lactamase of Pseudomonas aeruginosa (UniProt:Q932Y6, ChEMBL:CHEMBL1293246).
- App built in `Python` and `Streamlit` referring to the Chanin Nantasenamat's model (https://github.com/dataprofessor/bioactivity-prediction-app).
- Calculate Morgan-Descriptor by 'Rdkit' (http://www.rdkit.org/docs/GettingStartedInPython.html#morgan-fingerprints-circular-fingerprints).
---
""")

# Sidebar
with st.sidebar.header('Upload your CSV data'):
    uploaded_file = st.sidebar.file_uploader("Upload your input file", type=['txt'])
    st.sidebar.markdown("""
[Example input file](example.txt)
""")

if st.sidebar.button('Predict'):
    load_data = pd.read_table(uploaded_file, sep=' ', header=None)

    st.header('**Original input data**')
    st.write(load_data)

    with st.spinner("Calculating descriptors..."):
        desc_calc(load_data)

    # Read in calculated descriptors and display the dataframe
    st.header('**Calculated molecular descriptors**')
    desc = pd.read_csv('descriptor_output.csv')
    st.write(desc)
    st.write(desc.shape)

    # Read descriptor list used in previously built model
    st.header('**Subset of descriptors from previously built models**')
    Xlist = list(pd.read_csv('descriptor_list.csv').columns)
    desc_subset = desc[Xlist]
    st.write(desc_subset)
    st.write(desc_subset.shape)

# Apply trained model to make prediction on query compounds
    build_model(desc_subset)
else:
    st.info('Upload input data in the sidebar to start!')
