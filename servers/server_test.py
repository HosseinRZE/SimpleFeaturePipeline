import sys
from pathlib import Path
import glob
import joblib
import torch
import os
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from typing import Dict, List, Any
from utils.decorators.trace import trace
from torch.utils.data import DataLoader
from utils.padding_batch_reg import collate_batch
# Import necessary classes
# Note: LSTMKernelAttentionLSTMMultiRegressor needs to be importable
from models.neural_nets.vanilla_fnn import VanillaFNN
# Assuming the updated FeaturePipeline class and DataStoreMock are available
# Replace 'ServerPreprocess' import with 'DataStoreMock'
from servers.pre_process.data_store_mock import DataStoreMock # Assuming DataStoreMock is in a separate file or defined above

app = Flask(__name__)

# ---------------- Load model and meta ----------------
meta_files = glob.glob("/home/iatell/projects/meta-learning/experiments/fnn_train_model_20251107_163052/meta_train_model_20251107_163052.pkl")
state_files = glob.glob("/home/iatell/projects/meta-learning/experiments/fnn_train_model_20251107_163052/model_train_model_20251107_163052.pt")
pipeline_path = "/home/iatell/projects/meta-learning/experiments/fnn_train_model_20251107_163052/pipeline_train_model_20251107_163052.pkl"

# Pick the newest (last modified)
meta_path = max(meta_files, key=os.path.getmtime)
state_path = max(state_files, key=os.path.getmtime)
print("Using latest meta file:", meta_path)
meta = joblib.load(meta_path)

# Load the model
model = VanillaFNN.load_from_checkpoint(state_path)
model.eval()

# ---------------- Load raw data ----------------
df = pd.read_csv("/home/iatell/projects/meta-learning/data/Bitcoin_BTCUSDT_kaggle_1D_candles.csv", parse_dates=['timestamp'])
dense_df = df.set_index('timestamp').asfreq('D').ffill()

# ---------------- Setup pipeline ----------------
print(f"Loading full pipeline from: {pipeline_path}")
feature_pipeline = joblib.load(pipeline_path)
# ---------------- Initialize Data Store Mock ----------------
data_store_mock = DataStoreMock(initial_df=dense_df)
# ---------------- Run on_server_init hook ----------------
# Prepare state for the initial run
state = {
    "model": model, 
    "full_data": dense_df, 
    # Use the new data store object
    "data_store": data_store_mock 
}
# Execute the initialization hook
# Note: In a real app, any essential objects like the DataStoreMock 
# should be retrieved and stored globally if modified by the hook.
# For simplicity, we keep the global 'data_store_mock'.

# ---------------- Routes ----------------
@app.route("/")
def home():
    return render_template("hungarian.html")

@app.route("/get_and_add_data")
def get_and_add_data():
    initial_seq_len = 21
    next_idx = request.args.get("idx", type=int)
    
    if next_idx is None:
        # Initial request: Load the starting sequence and run 'on_first_request'
        if len(data_store_mock) == 0:
            initial_data = dense_df.iloc[:initial_seq_len]
            
            # Run the hook before adding data to the mock 
            # (allowing add-ons to modify initial_data or state if needed)
            state = {"initial_data": initial_data, "initial_seq_len": initial_seq_len, "data_store": data_store_mock}
            initial_data_to_add = state.get("initial_data", initial_data) # Use potentially modified data

            # Add data to the mock
            for _, row in initial_data_to_add.iterrows():
                data_store_mock.add_candle(row)
        
        # Candles to send to the client (from the original source for the initial view)
        candles = [{'time': int(ts.timestamp()),
                    'open': float(row.open),
                    'high': float(row.high),
                    'low': float(row.low),
                    'close': float(row.close)}
                    for ts, row in dense_df.iloc[:initial_seq_len].iterrows()]
        
        return jsonify({
            "initial_seq_len": initial_seq_len,
            "next_idx": initial_seq_len,
            "candles": candles
        })
    else:
        # Subsequent requests: Add one candle
        if next_idx >= len(dense_df):
            return jsonify({"error": "End of data"}), 404
            
        row = dense_df.iloc[next_idx]
        candle = {'time': int(row.name.timestamp()),
                  'open': float(row.open),
                  'high': float(row.high),
                  'low': float(row.low),
                  'close': float(row.close)}
                  
        # Add the new candle to the mock data store
        data_store_mock.add_candle(row) 
        
        return jsonify({"next_idx": next_idx + 1, "candle": candle})
    
@app.route("/predict", methods=['POST'])
def predict():
    print("\n==== /predict called ====")
    data = request.get_json(force=True)
    print(f"🟦 Received JSON: {data}")

    seq_len = data.get("seq_len")
    print(f"🟦 seq_len = {seq_len} (type: {type(seq_len)})")

    if not seq_len or not isinstance(seq_len, int):
        print("❌ Invalid seq_len format.")
        return jsonify({"error": "Provide 'seq_len' as an int"}), 400

    # 1. Run on_server_request hook
    print("➡️ Running on_server_request...")
    state = {
        "seq_len": seq_len,
        "data_store": data_store_mock,
        "df_data": data_store_mock.current_data
    }
    print(f"🟦 Current data_store length: {len(data_store_mock)}")
    
    state = feature_pipeline.run_on_server_request(state, feature_pipeline.extra_info)
        # 2. Create a DataLoader using the dataset AND your collate_fn
    inference_loader = DataLoader(
        dataset=state,
        batch_size=1,  # Or any batch size your server handles
        shuffle=False, # No need to shuffle for inference
        collate_fn=collate_batch 
    )
    model.eval() # Set model to evaluation mode
    predictions_list = []
    with torch.no_grad(): # Disable gradient computation
        for batch in inference_loader:
            # The collate_fn formats the data
            X_batch, y_dummy_batch, lengths_batch, _ = batch
            # (Optional) Move to GPU if your model is on GPU
            # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            # X_batch = {k: v.to(device) for k, v in X_batch.items()}
            # lengths_batch = {k: v.to(device) for k, v in lengths_batch.items()}
            # model.to(device)
            # 4. Call your model
            predictions = model(X_batch, lengths_batch)

            predictions_list.append(predictions)
# Check if the list is empty before concatenation (good practice)
    if not predictions_list:
        print("❌ Inference loop produced an empty prediction list.")
        return jsonify({"error": "No data received for prediction"}), 500
        
    y_pred_np = torch.cat(predictions_list, dim=0).cpu().numpy()
    last_close = float(data_store_mock.current_data['close'].iloc[-1])
    inference_payload = {
        "y_pred_np": y_pred_np,
        "last_close_price": last_close
    }
    inference_payload = feature_pipeline.run_on_server_inference(inference_payload, feature_pipeline.extra_info)
    scaled_pred_prices = inference_payload["y_pred_np"]   # 4. Return results
    print(f"🟩 Prediction successful: {scaled_pred_prices}")
    return jsonify({
            "pred_prices": scaled_pred_prices.tolist()
        })
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)