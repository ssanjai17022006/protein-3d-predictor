import streamlit as st
import torch
import torch.nn as nn
import pandas as pd
import os
import streamlit.components.v1 as components

# --- 1. Set Page Configuration ---
st.set_page_config(
    page_title="Adaptive Protein 3D Predictor",
    page_icon="🧬",
    layout="wide"
)

# --- 2. Define the Model Architecture (Must match training structure) ---
class ProteinLSTM(nn.Module):
    def __init__(self, vocab_size=21, embed_dim=32, hidden_dim=64, output_dim=3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        embedded = self.embed(x)
        lstm_out, _ = self.lstm(embedded)
        logits = self.fc(lstm_out)
        return logits

# --- 3. Dictionary Mappings & Constants ---
MAX_LEN = 128
AMINO_ACIDS = ["PAD", "A", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "W", "Y"]
AA_TO_IDX = {aa: idx for idx, aa in enumerate(AMINO_ACIDS)}
IDX_TO_STRUCT = {0: "H", 1: "E", 2: "C"}  # 0: Helix, 1: Sheet, 2: Coil

# --- 4. Cached Model Loading Function ---
@st.cache_resource
def load_trained_model():
    model = ProteinLSTM()
    model_path = "protein_lstm_model.pth"
    if os.path.exists(model_path):
        # Using weights_only=True for security and compatibility mapping to CPU
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu'), weights_only=True))
        model.eval()
        return model
    else:
        st.error(f"🚨 Critical Error: '{model_path}' weight binary not found on server!")
        return None

# --- 5. Helper Function: Tokenize and Pad Sequence ---
def preprocess_sequence(seq):
    seq = seq.upper().strip()
    tokens = [AA_TO_IDX.get(aa, 0) for aa in seq][:MAX_LEN]
    padded_tokens = tokens + [0] * (MAX_LEN - len(tokens))
    return torch.tensor([padded_tokens], dtype=torch.long), len(seq)

# --- 6. Helper Function: Render Interactive 3Dmol Mesh with Hover Labels ---
def render_3d_mesh(pdb_filename):
    if not os.path.exists(pdb_filename):
        return f"<p style='color:red;'>Error: Template {pdb_filename} missing from path.</p>"
    
    with open(pdb_filename, "r") as f:
        pdb_data = f.read().replace("\n", "\\n").replace("'", "\\'")
        
    html_content = f"""
    <div id="container" style="height: 450px; width: 100%; position: relative;"></div>
    <script src="https://code.jquery.com/jquery-3.6.3.min.js"></script>
    <script src="https://3dmol.org/build/3Dmol-min.js"></script>
    <script>
        $(document).ready(function() {{
            let element = $('#container');
            let config = {{ backgroundColor: '#111111' }};
            let viewer = $3Dmol.createViewer(element, config);
            viewer.addModel('{pdb_data}', "pdb");
            
            // Apply biological custom structural color map rules
            viewer.setStyle({{}}, {{cartoon: {{color: 'spectrum'}}}});
            viewer.zoomTo();
            viewer.render();

            // 🧠 INTERACTIVE HOVER INTERFACE (YOUR UPGRADE)
            viewer.setHoverable({{}}, true,
                function(atom, viewer, event, container) {{
                    if(!atom.label) {{
                        // Displays: [Residue Name] [Residue ID Number] (Atom Element)
                        atom.label = viewer.addLabel(
                            atom.resn + " " + atom.resi + " (" + atom.elem + ")", 
                            {{
                                position: atom, 
                                backgroundColor: '#1e1e1e', 
                                fontColor: '#ffffff',
                                borderSize: 1,
                                borderColor: '#ff4b4b',
                                backgroundOpacity: 0.85
                            }}
                        );
                    }}
                }},
                function(atom, viewer) {{
                    if(atom.label) {{
                        viewer.removeLabel(atom.label);
                        delete atom.label;
                    }}
                }}
            );
        }});
    </script>
    """
    return html_content

# --- 7. Application User Interface (UI Layout) ---
st.title("🧬 Adaptive Protein 3D Predictor")
st.markdown("---")

# Layout Configuration: Split Screen setup
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📥 Sequence Input Processing")
    default_seq = "MVLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKY"
    user_input = st.text_area("Enter Amino Acid Sequence (Max 128 Res):", value=default_seq, height=150)
    
    predict_btn = st.button("🚀 Predict & Compute Structure")

# Process Logic upon trigger
if predict_btn and user_input:
    clean_input = user_input.upper().strip()
    
    if len(clean_input) > MAX_LEN:
        st.error(f"⚠️ Sequence length ({len(clean_input)}) exceeds maximum permitted layer size of {MAX_LEN} characters.")
    else:
        # Load AI Brain
        model = load_trained_model()
        
        if model:
            # Inference Calculations
            input_tensor, dynamic_len = preprocess_sequence(clean_input)
            with torch.no_grad():
                logits = model(input_tensor)
                predictions = torch.argmax(logits, dim=-1).squeeze(0).numpy()
            
            # Trim padding elements
            real_predictions = predictions[:dynamic_len]
            struct_tags = [IDX_TO_STRUCT[p] for p in real_predictions]
            
            # Calculate Structural Composition Ratio
            helix_pct = struct_tags.count("H") / dynamic_len
            sheet_pct = struct_tags.count("E") / dynamic_len
            
            # Determine target homology structural template name based on structural dominance
            if helix_pct > 0.45:
                template_file = "1AIE.pdb"
            elif sheet_pct > 0.40:
                template_file = "1EMA.pdb"
            else:
                template_file = "1GFL.pdb"
                
            # Render Visual Layout Results
            with col1:
                st.markdown("### 📊 Predicted Secondary Architecture Matrix")
                res_df = pd.DataFrame({
                    "Residue Index": range(1, dynamic_len + 1),
                    "Amino Acid": list(clean_input),
                    "Predicted Target Shape": struct_tags
                })
                st.dataframe(res_df, height=250, use_container_width=True)
            
            with col2:
                st.subheader("🔮 Simulated 3D Morphological View")
                st.caption("✨ *Tip: Hover your mouse cursor directly over parts of the structure model to inspect atom data and residue identifiers.*")
                
                # High Visibility Structural Classification Labels
                if template_file == "1AIE.pdb":
                    st.success("🧬 **Active Structural Template: Human Villin Headpiece (Alpha-Helix Dominant)**")
                    st.caption("This conformation represents tight, right-handed spiral structural matrices structural folding patterns.")
                elif template_file == "1EMA.pdb":
                    st.info("🧬 **Active Structural Template: Enhanced Green Fluorescent Protein Core (Beta-Sheet Dominant)**")
                    st.caption("This conformation represents anti-parallel, rigid pleated sheet structural scaffolding profiles.")
                else:
                    st.warning("🧬 **Active Structural Template: Ubiquitin Structural Variant (Mixed Alpha/Beta Conformation)**")
                    st.caption("This conformation represents a balanced physiological mix of secondary helices, sheets, and random coils.")
                
                # Render Graphic Frame canvas
                html_code = render_3d_mesh(template_file)
                components.html(html_code, height=480)
