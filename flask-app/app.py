import json
from flask import Flask, request
import make_palette

app = Flask(__name__)

@app.route("/")
def main():
    imgurl = request.args.get("url")
    if imgurl is None: return "No url specified"
    clusters = make_palette.url_to_clusters(imgurl)
    return json.dumps({"clusters":clusters})


if __name__ == "__main__":
    app.run(debug=True)
