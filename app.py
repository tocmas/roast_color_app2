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
    # ラベルに対応する名称を返すための関数を15クラスに対応するように修正
    if label == 0:
        return "Class 1"
    elif label == 1:
        return "Class 2"
    elif label == 2:
        return "Class 3"
    elif label == 3:
        return "Class 4"
    elif label == 4:
        return "Class 5"
    elif label == 5:
        return "Class 6"
    elif label == 6:
        return "Class 7"
    elif label == 7:
        return "Class 8"
    elif label == 8:
        return "Class 9"
    elif label == 9:
        return "Class 10"
    elif label == 10:
        return "Class 11"
    elif label == 11:
        return "Class 12"
    elif label == 12:
        return "Class 13"
    elif label == 13:
        return "Class 14"  
    elif label == 14:
        return "Class 15"
    
app = Flask(__name__)

ALLOWED_EXTENTIONS = set(["png", "jpg", "gif", "jpeg"])

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
            base64_data = "data:image/png;base64, {}".format(base64_str)
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
