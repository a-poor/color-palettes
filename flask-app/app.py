import json
from flask import Flask, request
import make_palette

app = Flask(__name__)

def rgb2hex(r,g,b):
    return f"{r:02x}{g:02x}{b:02x}"

def make_palette_url(palette):
    return "https://coolors.co/" + "-".join(rgb2hex(*c) for c in (palette))

@app.route("/")
def main():
    imgurl = request.args.get("url")
    if imgurl is None: return "No url specified"
    clusters = make_palette.url_to_clusters(imgurl)
    return json.dumps({"clusters":clusters})

@app.route("/palette-url")
def palette_url():
    imgurl = request.args.get("url")
    if imgurl is None: return "No url specified"
    clusters = make_palette.url_to_clusters(imgurl)
    return json.dumps({"palette_url":make_palette_url(clusters)})


if __name__ == "__main__":
    app.run(debug=True)
