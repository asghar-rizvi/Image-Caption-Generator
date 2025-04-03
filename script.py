from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Model, load_model
import os


def store_vgg16():
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)

    # Load VGG16 (will download weights on first run)
    base_model = VGG16(weights='imagenet')

    # Extract only the feature extraction layers
    feature_extractor = Model(
        inputs=base_model.inputs,
        outputs=base_model.layers[-2].output  # Last layer before classification
    )

    # Save the feature extractor locally
    feature_extractor.save('model/vgg16_feature_extractor.keras')

    print("VGG16 feature extractor saved successfully!")
    
if __name__ == '__main__' :
    store_vgg16()