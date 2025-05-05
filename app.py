from flask import Flask, request, jsonify
import tensorflow as tf, numpy as np, cv2

# 3.1 Load .keras artifact
model = tf.keras.models.load_model('best_sports_classifier.keras')

# 3.2 Train-time normalization stats (update as needed)
TRAIN_MEAN = np.array([0.485,0.456,0.406])
TRAIN_STD  = np.array([0.229,0.224,0.225])

# 3.3 Class names in correct order
CLASS_NAMES = ['basketball','football','ping_pong','tennis','volleyball','baseball','hockey']

app = Flask(__name__, static_folder='static')

@app.route('/health')
def health():
    return 'OK', 200

def preprocess_image(img_bytes):
    arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (64,64))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.0
    return (img - TRAIN_MEAN)/TRAIN_STD

@app.route('/predict', methods=['POST'])
def predict():
    files = request.files.getlist('images')
    if not files:
        return jsonify({'error':'No images uploaded'}),400
    batch = np.stack([preprocess_image(f.read()) for f in files], axis=0)
    preds = model.predict(batch)
    idxs  = preds.argmax(axis=1)
    labels = [CLASS_NAMES[i] for i in idxs]
    return jsonify({'predictions': labels})

@app.route('/')
def home():
    return app.send_static_file('index.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
