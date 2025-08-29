from flask import Flask, request, jsonify, render_template
import pandas as pd
from labeling.scripts.talib_candle import CandleLabeler  # import the class we fixed

app = Flask(__name__)
labeler = CandleLabeler()

@app.route("/")
def home():
    return render_template("ta_lib_candle.html")  # your HTML file

@app.route("/label", methods=["POST"])
def label_csv():
    file = request.files["file"]
    df = pd.read_csv(file)
    try:
        df_labeled = labeler.label_dataframe(df)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400

    # Convert label lists to JSON arrays
    result = df_labeled.to_dict(orient="records")
    return jsonify(result)

@app.route("/legend")
def get_legend():
    return jsonify(labeler.get_legend())

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)