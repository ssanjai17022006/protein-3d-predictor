import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import streamlit.components.v1 as components
import os

# --- 1. MODEL ARCHITECTURE ---
class ProteinLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(21, 32, padding_idx=0)
        self.lstm = nn.LSTM(32, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(128, 3)

    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

# --- 2. LOCAL 3D RENDERER ---
def render_local_3d(pdb_file_path):
    if not os.path.exists(pdb_file_path):
        return st.error(f"File {pdb_file_path} not found in directory!")
    
    with open(pdb_file_path, "r") as f:
        pdb_data = f.read().replace("\n", "\\n").replace("'", "\\'")

    html_code = f"""
    <script src="https://code.jquery.com/jquery-3.6.3.min.js"></script>
    <script src="https://3Dmol.org/build/3Dmol-min.js"></script>
    <div id="container" style="height: 500px; width: 100%; position: relative; border-radius:10px; border:1px solid #ddd;"></div>
    <script>
        $(function() {{
            let element = $('#container');
            let viewer = $3Dmol.createViewer(element, {{ backgroundColor: 'white' }});
            viewer.addModel('{pdb_data}', "pdb");
            viewer.setStyle({{}}, {{cartoon: {{colorfunc: function(atom) {{
                if (atom.ss === 'h') return '#FF0000'; // Red
                if (atom.ss === 's') return '#FFFF00'; // Yellow
                return '#00FF00';                      // Green
            }}}}}});
            viewer.zoomTo();
            viewer.render();
        }});
    </script>
    """
    return components.html(html_code, height=520)

# --- 3. THE APP ---
st.set_page_config(page_title="Dynamic Bio-AI", layout="wide")
st.title("🧬 Adaptive Protein 3D Predictor")

@st.cache_resource
def load_model():
    m = ProteinLSTM()
    m.load_state_dict(torch.load("protein_lstm_model.pth"))
    m.eval()
    return m

col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Sequence Input")
    sequence = st.text_area("Amino Acid Sequence:", "MVLSEGEW...", height=150)
    
    if st.button("Analyze & Fold"):
        model = load_model()
        # Prediction Logic
        AA_DICT = {res: i+1 for i, res in enumerate("ACDEFGHIKLMNPQRSTVWY")}
        seq_ids = torch.tensor([[AA_DICT.get(aa, 0) for aa in sequence.upper()]])
        with torch.no_grad():
            output = model(seq_ids)
            preds = torch.argmax(output, dim=2)[0].tolist()
        
        # Determine the "Winner" (Most common structure)
        h_count = preds.count(0)
        e_count = preds.count(1)
        
        if h_count > e_count:
            st.session_state.pdb_choice = "1AIE.pdb"
            st.info("Detecting Alpha-Helix Dominance...")
        elif e_count > h_count:
            st.session_state.pdb_choice = "1EMA.pdb"
            st.info("Detecting Beta-Sheet Dominance...")
        else:
            st.session_state.pdb_choice = "1GFL.pdb"

with col2:
    st.subheader("Predicted 3D Structure")
    current_pdb = st.session_state.get('pdb_choice', '1AIE.pdb')
    render_local_3d(current_pdb)
    st.write(f"Showing Template: **{current_pdb}**")