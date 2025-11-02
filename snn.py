import os
import random
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Lambda, MaxPooling2D, Subtract, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import backend as K
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, classification_report
from scipy.spatial.distance import euclidean
import warnings
warnings.filterwarnings('ignore')

# Path to your dataset
DATASET_PATH = "C:/Users/karti/Code/projects/infant-cry-classification/dataset/"

# Categories (Ensure these match your actual folder names)
CATEGORIES = ["belly_pain", "burping", "hungry", "discomfort", "tired"]

# Fixed Audio Length (7 sec)
FIXED_DURATION = 7.0
SAMPLE_RATE = 22050  # Standard sampling rate
TIMESTEPS = 300  # Fixed MFCC time frames

# Enhanced feature extraction with multiple audio features
def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
        
        # Adjust to exactly 7 seconds
        target_length = int(FIXED_DURATION * SAMPLE_RATE)
        if len(y) > target_length:
            y = y[:target_length]  # Trim
        elif len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)))  # Pad with zeros

        # Extract multiple features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40, n_fft=2048, hop_length=512)
        mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=2048, hop_length=512)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512)
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=2048, hop_length=512)
        
        # Convert mel spectrogram to log scale
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Stack all features
        features = np.vstack([mfcc, log_mel, chroma, spectral_contrast])
        
        # Normalize features
        features = (features - np.mean(features)) / (np.std(features) + 1e-8)

        # Resize to fixed shape
        if features.shape[1] < TIMESTEPS:
            features = np.pad(features, ((0, 0), (0, TIMESTEPS - features.shape[1])), mode='constant')
        else:
            features = features[:, :TIMESTEPS]

        # Reshape to fit CNN (add channel dimension)
        return np.expand_dims(features, axis=-1)  # Shape: (feature_dim, TIMESTEPS, 1)
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to visualize MFCC spectrogram
def plot_mfcc(file_path):
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

    plt.figure(figsize=(10, 4))
    librosa.display.specshow(mfcc, sr=sr, x_axis="time", cmap="coolwarm")
    plt.colorbar()
    plt.title("MFCC Visualization")
    plt.show()

# Load and preprocess dataset with data validation
def load_dataset():
    X, Y = [], []
    for category in CATEGORIES:
        category_path = os.path.join(DATASET_PATH, category)
        if not os.path.exists(category_path):
            print(f"⚠️ Warning: Category folder '{category}' not found!")
            continue

        files = [f for f in os.listdir(category_path) if f.endswith(".wav")]
        if not files:
            print(f"⚠️ No .wav files in '{category_path}'!")

        for file in files:
            file_path = os.path.join(category_path, file)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                Y.append(category)

    print(f"✅ Loaded {len(X)} samples from {len(CATEGORIES)} categories.")
    
    # Check class distribution
    unique, counts = np.unique(Y, return_counts=True)
    print("Class distribution:", dict(zip(unique, counts)))
    
    return np.array(X), np.array(Y)

# Enhanced Siamese Network Model with better architecture
def build_siamese_model(input_shape):
    input_layer = Input(shape=input_shape)
    
    # First Conv Block
    x = Conv2D(32, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(input_layer)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Second Conv Block
    x = Conv2D(64, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.2)(x)
    
    # Third Conv Block
    x = Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Dropout(0.3)(x)
    
    x = Flatten()(x)
    x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu', kernel_regularizer=l2(0.001))(x)
    
    return Model(input_layer, x)

# Build full Siamese network with contrastive loss
def build_siamese_network(input_shape):
    base_network = build_siamese_model(input_shape)

    input_a = Input(shape=input_shape)
    input_b = Input(shape=input_shape)

    # Get feature embeddings
    encoded_a = base_network(input_a)
    encoded_b = base_network(input_b)

    # Compute L1 distance
    l1_distance = Lambda(lambda tensors: K.abs(tensors[0] - tensors[1]))([encoded_a, encoded_b])
    output = Dense(1, activation='sigmoid')(l1_distance)

    return Model(inputs=[input_a, input_b], outputs=output)

# Improved pair generation with balanced sampling
def create_pairs(X, Y):
    pairs, labels = [], []
    class_indices = {label: np.where(Y == label)[0] for label in np.unique(Y)}
    
    # Ensure balanced pairs
    min_samples = min(len(indices) for indices in class_indices.values())
    
    for idx in range(len(X)):
        current_class = Y[idx]
        
        # Positive pair
        pos_indices = class_indices[current_class]
        pos_idx = random.choice(pos_indices)
        pairs.append([X[idx], X[pos_idx]])
        labels.append(1)  # Similar
        
        # Negative pair - choose from different class
        neg_classes = [c for c in np.unique(Y) if c != current_class]
        neg_class = random.choice(neg_classes)
        neg_idx = random.choice(class_indices[neg_class])
        pairs.append([X[idx], X[neg_idx]])
        labels.append(0)  # Dissimilar

    return np.array(pairs), np.array(labels)

# Contrastive loss function for better similarity learning
def contrastive_loss(y_true, y_pred, margin=1.0):
    square_pred = K.square(y_pred)
    margin_square = K.square(K.maximum(margin - y_pred, 0))
    return K.mean(y_true * square_pred + (1 - y_true) * margin_square)

# Load and process dataset
print("Loading dataset...")
X, Y = load_dataset()

# Ensure dataset is loaded
if len(X) == 0:
    print("❌ No data loaded! Check dataset path and files.")
    exit()

# Encode labels as numerical values
encoder = LabelEncoder()
Y_numeric = encoder.fit_transform(Y)

# Split dataset into train and test with stratification
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y_numeric, test_size=0.2, random_state=42, stratify=Y_numeric
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Generate pairs for training and testing
print("Generating training pairs...")
train_pairs, train_labels = create_pairs(X_train, Y_train)
print("Generating testing pairs...")
test_pairs, test_labels = create_pairs(X_test, Y_test)

print(f"Training pairs: {len(train_pairs)}")
print(f"Testing pairs: {len(test_pairs)}")

# Build SNN model - Update input shape based on new feature dimension
# With stacked features: MFCC(40) + Mel(40) + Chroma(12) + Spectral Contrast(7) = 99 features
input_shape = (99, TIMESTEPS, 1)  # Updated shape for stacked features
siamese_model = build_siamese_network(input_shape)

# Compile model with lower learning rate
siamese_model.compile(
    optimizer=Adam(learning_rate=0.0001), 
    loss="binary_crossentropy", 
    metrics=["accuracy"]
)

# Add callbacks for better training
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7),
    EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
]

# Train model with validation split
print("Starting training...")
history = siamese_model.fit(
    [train_pairs[:, 0], train_pairs[:, 1]], 
    train_labels, 
    batch_size=32,
    epochs=50,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

# Evaluate model
print("Evaluating model...")
y_pred_prob = siamese_model.predict([test_pairs[:, 0], test_pairs[:, 1]])
y_pred = np.round(y_pred_prob)  # Convert probabilities to binary labels

accuracy = accuracy_score(test_labels, y_pred)
precision = precision_score(test_labels, y_pred)

print(f"✅ Model Accuracy: {accuracy:.4f}")
print(f"✅ Model Precision: {precision:.4f}")

# Enhanced prediction function using the trained Siamese network
def predict_category(siamese_model, test_feature, X_train, Y_train, encoder, num_matches=5):
    """Predict category using similarity scores from Siamese network"""
    similarities = []
    
    for train_feature, train_label in zip(X_train, Y_train):
        # Create pair and get similarity score
        pair = np.array([[test_feature, train_feature]])
        similarity_score = siamese_model.predict([pair[:, 0], pair[:, 1]], verbose=0)[0][0]
        similarities.append((similarity_score, train_label))
    
    # Sort by similarity score (highest first)
    similarities.sort(key=lambda x: x[0], reverse=True)
    
    # Get top matches
    top_matches = similarities[:num_matches]
    
    # Count votes from top matches
    vote_count = {}
    for score, label in top_matches:
        vote_count[label] = vote_count.get(label, 0) + 1
    
    # Get predicted label with most votes
    predicted_label = max(vote_count, key=vote_count.get)
    
    return predicted_label, top_matches

# Test with a new audio file
test_file = "C:/Users/karti/Code/projects/infant-cry-classification/test.wav"
if os.path.exists(test_file):
    test_feature = extract_features(test_file)
    
    if test_feature is not None:
        # Use the enhanced prediction function
        predicted_label, top_matches = predict_category(siamese_model, test_feature, X_train, Y_train, encoder)
        
        print(f"Predicted Category: {encoder.inverse_transform([predicted_label])[0]}")
        print("Top 5 matches:")
        for i, (score, label) in enumerate(top_matches, 1):
            print(f"  {i}. {encoder.inverse_transform([label])[0]} (similarity: {score:.4f})")
        
        # Visualize MFCC of test file
        plot_mfcc(test_file)
    else:
        print("❌ Failed to extract features from test file")
else:
    print(f"❌ Test file not found: {test_file}")

# Plot training history
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.show()
