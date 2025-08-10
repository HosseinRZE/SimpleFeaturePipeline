# server_transformer.py
import glob
import joblib
import torch
from flask import Flask, request, jsonify, render_template
from preprocess import load_raw_data_serve, one_sample
from models.transformer.transform_classifier import TransformerClassifier

app = Flask(__name__)

meta_path = glob.glob("models/saved_models/transformer_meta.pkl")[0]
state_path = glob.glob("models/saved_models/transformer.ckpt")[0]
meta = joblib.load(meta_path)

model = TransformerClassifier(
    input_dim=meta['input_dim'],
    seq_len=meta['seq_len'],
    d_model=meta['d_model'],
    num_heads=meta['num_heads'],
    num_layers=meta['num_layers'],
    num_classes=meta['num_classes'],
    lr=meta['lr']
)
model.load_state_dict(torch.load(state_path, map_location='cpu')["state_dict"])
model.eval()

df = load_raw_data_serve("data/candles.csv", "data/labels.csv")

@app.route("/predict", methods=['POST'])
def predict():
    idx = request.json['idx']
    seq_df = df.iloc[idx - meta['seq_len'] + 1: idx + 1]
    X_np = one_sample(seq_df)
    X_t = torch.tensor(X_np, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        logits = model(X_t)
        pred_class = torch.argmax(logits, dim=1).item()
    return jsonify({"class": int(pred_class)})

if __name__ == "__main__":
    app.run(debug=True)
