from flask import request, render_template, Flask, redirect, url_for, flash
from utils import Model, predict
from sentence_transformers import SentenceTransformer
import torch as t

app = Flask(__name__)
embedding_model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
model = Model(embedding_dim=embedding_model.get_sentence_embedding_dimension())
model.load_state_dict(t.load("./model_20240921-125240.pth", map_location="cpu"))

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None
    if request.method == "POST":

        # Get the user email
        email_contents = request.form["input-email"]
        
        # Make the prediction from the model
        prediction = predict(
            model=model,
            embedding_model=embedding_model,
            text=email_contents
        )
        
    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)