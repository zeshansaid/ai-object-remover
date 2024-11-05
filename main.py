from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import torch
import base64
import torchvision as tv
from PIL import Image, ImageFile
from objectRemoval_engine import SimpleLama
from PIL import Image
import io
import cv2
import numpy as np
from projectUtils import *
ImageFile.LOAD_TRUNCATED_IMAGES = True


app = Flask(__name__)
app.config['SECRET_KEY'] = 'vnkdjnfjknfl1232#'
CORS(app)
app.app_context().push()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
simple_lama = SimpleLama(device=device)
# -------------------------------------------- ------------------ -----------------------------------------
# -------------------------------------------- index ------------------------------------------------------
# -------------------------------------------- ------------------ -----------------------------------------
@app.route('/')
def hello_world():
    return "api , online"

 

#------------------------------------ Object removal ---------------------------------------------

"""
[{'startX': 1033.9316, 'startY': 1210.915, 'endX': 1033.9316, 'endY': 1267.9419, 'strokeWidth': 20}, {'startX': 1033.9316, 'startY': 1267.9419, 'endX': 1033.9316, 'endY': 1272.8931, 'strokeWidth': 20}, {'startX': 1033.9316, 'startY': 1272.8931, 'endX': 1033.9316, 'endY': 1288.873, 'strokeWidth': 20}, {'startX': 1033.9316, 'startY': 1288.873, 'endX': 1033.9316, 'endY': 1293.9355, 'strokeWidth': 20}]
"""

@app.route('/removeobj', methods=['POST'])
def object_removal():
    print("/REMOVE_OBJ new request coming")
    data = request.get_json()
    base64Image= data["image"]
    base64mask= data["mask"]
    size = data["size"]

    print(f"Size found : {size}")

    cv_img = base64toopencv(base64Image)
    cv_mask = base64toopencv(base64mask)
    cv2.imwrite("test.png",cv_img)
    h, w, c = cv_img.shape
    cv_mask = cv2.resize(cv_mask, (w, h))
    cv2.imwrite("test_mask.png",cv_mask)


    cv_mask = cv2.cvtColor(cv_mask, cv2.COLOR_BGR2RGB) 
    cv_mask = Image.fromarray(cv_mask).convert('L') 

    cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) 
    cv_img = Image.fromarray(cv_img) 

    result = simple_lama(cv_img, cv_mask)
    new_image = result
    bio = io.BytesIO()
    new_image.save(bio, "PNG")
    bio.seek(0)
    im_b64 = base64.b64encode(bio.getvalue()).decode()

    return jsonify({"bg_image":im_b64}) # end of function end point removeobj 


 
if __name__ == '__main__':
    # app.run(debug=True, use_reloader=False)
    app.run(host="0.0.0.0", port=5000, debug=True,use_reloader=False)
