import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.models import load_model
import pickle
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
MAX_LENGTH = 37

def load_models():
    global VGG_MODEL, caption_model, tokenizer
    
    VGG_MODEL = load_model('model/vgg16_feature_extractor.keras')

    caption_model = load_model('model/Vision2Text.keras')

    with open('model/tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    print(len(tokenizer.word_index))

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length, padding='post')
        yhat = model.predict([image, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = idx_to_word(yhat, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == 'endseq':
            break

    in_text = in_text.replace('startseq', '')
    in_text = in_text.replace('endseq', '')
    return in_text

def preprocess_img(img_path):
    image = load_img(img_path, target_size=(224, 224))
    image = img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = preprocess_input(image)

    feature = VGG_MODEL.predict(image, verbose=0)
    caption = predict_caption(caption_model, feature, tokenizer, MAX_LENGTH)
    return caption
    
if __name__ == '__main__':
    load_models()
    test_image = '2_dogs.jpg'  
    caption = preprocess_img(test_image)
    print(f"\nGenerated caption: {caption}")