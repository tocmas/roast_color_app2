import torch
from color import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load("./roast_color_app3.pt", map_location=torch.device("cpu")))
    img = transform(img)
    img = img.unsqueeze(0)
    y = torch.argmax(net(img), dim=1).cpu().detach().item()
    return y

def getName(label):
    class_names = [
        "Very light roast",
        "Light roast",
        "Light dark roast",
        "Medium light roast",
        "Medium roast",
        "Medium dark roast",
        "Dark roast",
        "Very dark roast"
    ]
    return class_names[label]
    
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(["png", "jpg", "gif", "jpeg"])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def predicts():
    if request.method == "POST":
        if "filename" not in request.files:
            print("No 'filename' in uploaded files")
            return redirect(request.url)

        file = request.files["filename"]
        if file and allowed_file(file.filename):
            buf = io.BytesIO()
            image = Image.open(file) 
            image.save(buf, "png")
            base64_str = base64.b64encode(buf.getvalue()).decode("utf-8")
            base64_data = "data:image/png;base64,{}".format(base64_str)
            pred = predict(image)
            colorName_= getName(pred)
            return render_template("result.html", colorName=colorName_, image = base64_data)
        else:
            if not file:
                print("File is empty")
            elif not allowed_file(file.filename):
                print("File type not allowed")
            return redirect(request.url)
 
    elif request.method == "GET":
        return render_template("index.html")
    
if __name__ == "__main__":
    app.run(debug=True)
