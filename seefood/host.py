from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from nn import *
from parameters import *
import os

# os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Define a flask app and configure an upload directory
app = Flask('Seefood')
app.config['UPLOAD_FOLDER'] = './static/uploads'
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Construct model
model = construct_model(model_directory)

# Generates the HTML template that we have constructed
@app.route('/')
def hello():
    return render_template('home.html')

# Generate the prediction from the uploads folder
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['file']

        file_path = os.path.join(app.config['UPLOAD_FOLDER'],
                                 secure_filename(f.filename))
        f.save(file_path)

        preds = model_predict(file_path, model, target_size)
        result, score = decode(preds)
        generate_overlay(file_path, score, 'tick.png', 'cross.png', target_size)

        return render_template('result.html', result=result, url=file_path)
    return render_template('home.html')

@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False)
