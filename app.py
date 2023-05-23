import torch
from color import transform, Net
from flask import Flask, request, render_template, redirect
import io
from PIL import Image
import base64

def predict(img):
    net = Net().cpu().eval()
    net.load_state_dict(torch.load("I:\\roast_color_app\\roast_color_app.pt", map_location=torch.device("cpu")))
    img = transform(img)
    img = img.unsqueeze(0)
    y = torch.argmax(net(img), dim=1).cpu().detach().numpy()
    return y

def getName(label):
    class_names = [
        "Class 1", "Class 2", "Class 3", "Class 4", "Class 5",
        "Class 6", "Class 7", "Class 8", "Class 9", "Class 10",
        "Class 11", "Class 12", "Class 13", "Class 14", "Class 15"
    ]
    return class_names[label]
    
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(["png", "jpg", "gif", "jpeg"])

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENTIONS

@app.route("/", methods=["GET", "POST"])
def predicts():
    if request.method == "POST":
        if "filename" not in request.files:
            print("No 'filename' in uploaded files")
            return redirect(request.url)

        file = request.files["filename"]
        if file and allowed_file(file.filename):
            buf = io.BytesIO() # Typoを修正（BytestIO -> BytesIO）
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
