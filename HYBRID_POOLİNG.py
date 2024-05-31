#HYBRID POOLİNG

import numpy as np
import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import cv2
from keras.utils import to_categorical
from keras import models
from keras.layers import Conv2D, Input, MaxPooling2D, Flatten, Dense
from keras.optimizers import Adam
from sklearn.metrics import classification_report,confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.layers import Input, MaxPooling2D, Conv2D
from tensorflow.keras.layers import Concatenate, AveragePooling2D
from tensorflow.keras.models import Model


data_dir = '/content/drive/MyDrive/sondataset'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')
val_dir = os.path.join(data_dir, 'val')

norm_train_dir = os.path.join(train_dir, 'high_std', 'normal')
pn_train_dir = os.path.join(train_dir, 'high_std', 'pneumonia')

norm_test_dir = os.path.join(test_dir, 'NORMAL')
pn_test_dir = os.path.join(test_dir, 'PNEUMONIA')

norm_val_dir = os.path.join(val_dir, 'NORMAL')
pn_val_dir = os.path.join(val_dir, 'PNEUMONIA')


train_data = []

norm_imgs = glob.glob(f'{norm_train_dir}/*.jpeg')
pn_imgs = glob.glob(f'{pn_train_dir}/*.jpeg')

for img in norm_imgs:
    train_data.append((img,0))

for img in pn_imgs:
    train_data.append((img, 1))

train_data = pd.DataFrame(train_data, columns=['image_path', 'label'],index=None)

train_data = train_data.sample(frac = 1.0).reset_index(drop=True)

test_data = []

norm_imgs = glob.glob(f'{norm_test_dir}/*.jpeg')
pn_imgs = glob.glob(f'{pn_test_dir}/*.jpeg')

for img in norm_imgs:
    test_data.append((img,0))

for img in pn_imgs:
    test_data.append((img, 1))

test_data = pd.DataFrame(test_data, columns=['image_path', 'label'],index=None)
test_data = test_data.sample(frac = 1.0).reset_index(drop=True)

val_data = []

norm_imgs = glob.glob(f'{norm_val_dir}/*.jpeg')
pn_imgs = glob.glob(f'{pn_val_dir}/*.jpeg')

for img in norm_imgs:
    val_data.append((img,0))

for img in pn_imgs:
    val_data.append((img, 1))

val_data = pd.DataFrame(val_data, columns=['image_path', 'label'],index=None)
val_data = val_data.sample(frac = 1.0).reset_index(drop=True)

val_data.head()

train_counts = train_data.label.value_counts().to_dict()
test_counts = test_data.label.value_counts().to_dict()
val_counts = val_data.label.value_counts().to_dict()

fig,ax = plt.subplots(nrows=1,ncols=3,figsize=(10,5))

ax[0].pie(train_counts.values(),autopct="%1.2f%%")
ax[0].set_title('Train Data')
ax[0].legend(labels=train_counts.keys(),loc='upper right', bbox_to_anchor=(1.25,1), fontsize=8,frameon=False)

ax[1].pie(test_counts.values(),autopct="%1.2f%%")
ax[1].set_title('Test Data')
ax[1].legend(labels=test_counts.keys(),loc='upper right', bbox_to_anchor=(1.25,1), fontsize=8,frameon=False)

ax[2].pie(val_counts.values(),autopct="%1.2f%%")
ax[2].set_title('Validation Data')
ax[2].legend(labels=val_counts.keys(),loc='upper right', bbox_to_anchor=(1.25,1), fontsize=8,frameon=False)

plt.show()

total = len(train_data)
norm_train = len(train_data[train_data.label==0])
print(f"Total images in Training data: {total} of which {norm_train} are NORMAL and {total-norm_train} are PNEUMONIA.")

total = len(test_data)
norm_test = len(test_data[test_data.label==0])
print(f"Total images in Testing data: {total} of which {norm_test} are NORMAL and {total-norm_test} are PNEUMONIA.")

total = len(val_data)
norm_val = len(val_data[val_data.label==0])
print(f"Total images in Validation data: {total} of which {norm_val} are NORMAL and {total-norm_val} are PNEUMONIA.")

cat = {0:'NORMAL', 1:'PNEUMONIA'}
sample_norm_imgs = train_data.loc[train_data['label']==0].image_path.values.tolist()[:6]
sample_pn_imgs = train_data.loc[train_data['label']==1].image_path.values.tolist()[:6]

sample_imgs = sample_norm_imgs + sample_pn_imgs
fig, ax = plt.subplots(nrows=2, ncols=6, figsize=(15,5))
fig.suptitle('Sample Training Images')
for i in range(len(sample_imgs)):
    xc = i//6
    yc = i%6
    img = cv2.imread(sample_imgs[i])
    ax[xc,yc].imshow(img)
    ax[xc,yc].set_title(cat[xc])
    ax[xc,yc].set_xticks([])
    ax[xc,yc].set_yticks([])



def get_image_data(img_df):
    n = len(img_df)
    img_data = np.zeros((n,224,224))
    label_data = np.zeros((n,))

    for i in range(n):
        img_path = img_df.loc[i,"image_path"]
        img_label = img_df.loc[i,"label"]

        img_arr = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(img_arr, (224,224)) / 255

        img_data[i] = resized
        label_data[i] = img_label

    return img_data, label_data


X_train, y_train = get_image_data(train_data)
X_test, y_test = get_image_data(test_data)
X_val, y_val = get_image_data(val_data)


X_train = X_train.reshape(-1, 224, 224, 1)
X_test = X_test.reshape(-1, 224, 224, 1)
X_val = X_val.reshape(-1, 224, 224, 1)


y_train = to_categorical(y_train, num_classes=2)
y_val = to_categorical(y_val, num_classes=2)

print(X_train.shape, y_train.shape, sep='\n')
print(X_test.shape, y_test.shape, sep='\n')
print(X_val.shape, y_val.shape, sep='\n')


img_gen = ImageDataGenerator(rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode='nearest')

img_gen.fit(X_train)

train_generator = img_gen.flow(X_train, y_train)

type(train_generator)

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, AveragePooling2D, Concatenate, Lambda
from tensorflow.keras.models import Model

# Hibrit Pooling Fonksiyonu
def hybrid_pooling(pool_size=(2,2), alpha=0.5):
    def apply(x):
        max_pool = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='same')(x)
        avg_pool = AveragePooling2D(pool_size=pool_size, strides=(1, 1), padding='same')(x)
        weighted_avg = Lambda(lambda xx: xx * alpha)(avg_pool)
        weighted_max = Lambda(lambda xx: xx * (1 - alpha))(max_pool)
        return Concatenate()([weighted_avg, weighted_max])
    return apply

# Giriş Katmanı
inputs = Input(shape=(224, 224, 1))

# Evrişim Bloğu
conv_block = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(inputs)
conv_block = Conv2D(filters=64, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = hybrid_pooling()(conv_block)

conv_block = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=128, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = hybrid_pooling()(conv_block)

conv_block = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=256, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = hybrid_pooling()(conv_block)

conv_block = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = hybrid_pooling()(conv_block)

conv_block = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = Conv2D(filters=512, kernel_size=(3,3), activation='relu', padding='same')(conv_block)
conv_block = hybrid_pooling()(conv_block)

# Flatten Katmanı
flattened = Flatten()(conv_block)

# Çıkış Katmanı
outputs = Dense(2, activation='softmax')(flattened)

# Model Oluşturma
model = Model(inputs=inputs, outputs=outputs)

# Modeli Derleme
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Modeli Görüntüleme
model.summary()

conv_model = models.Model(inputs=inputs, outputs=conv_block)
model = models.Model(inputs=inputs, outputs=outputs)

model.summary()

for layer in conv_model.layers:
    if 'conv' in layer.name:
        filters, bias = layer.get_weights()
        print(layer.name, filters.shape)
    else:
        print(layer.name)


model.compile(optimizer=Adam(learning_rate=0.00001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

train_loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(train_loss) + 1)

plt.plot(epochs, train_loss, label='Training Loss')
plt.plot(epochs, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('vgg_model_High_Hybrid_VGG.h5')

from IPython.display import FileLink
FileLink(r'vgg_model_High_Hybrid_VGG.h5')

y_pred = model.predict(X_test)

y_true = y_test.astype(dtype=int)
y_pred = np.argmax(y_pred, axis=1)
print(classification_report(y_true, y_pred))

cm = confusion_matrix(y_true,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()