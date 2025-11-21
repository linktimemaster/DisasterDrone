import tensorflow as tf
from tensorflow.keras import datasets, layers, models, optimizers, regularizers, applications
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

base_model_MNv3 = applications.MobileNetV3Small(input_shape=(192,192,3), include_top=False, weights='imagenet', dropout_rate=0.2)
input = tf.keras.Input(shape=(192,192,3), name='input')
x = base_model_MNv3(input, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
output = tf.keras.layers.Dense(2)(x)
model_MNv3 = tf.keras.Model(inputs=input, outputs=output)

def process_image(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_image(img, channels=3, expand_animations=False)
    img = tf.cast(img, tf.float32) / 255.0
    return img

normal_paths = glob.glob(os.path.join("building_images/normal_building_images", "**", "*.png"), recursive=True)
broken_paths = glob.glob(os.path.join("building_images/broken_building_images", "**", "*.png"), recursive=True)

normal_set = [(process_image(i), 0) for i in normal_paths]
broken_set = [(process_image(i), 1) for i in broken_paths]

complete_set = normal_set + broken_set

np.random.shuffle(complete_set)

split = 25

train_x = np.array([i[0] for i in complete_set[:split]])
train_y = np.array([i[1] for i in complete_set[:split]])
val_x = np.array([i[0] for i in complete_set[split:]])
val_y = np.array([i[1] for i in complete_set[split:]])

model_MNv3.compile(optimizer=optimizers.Adam(learning_rate=0.000001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

finetune_model_MNv3 = model_MNv3.fit(train_x, train_y, epochs=10,
                    validation_data=(val_x, val_y))

# plt.plot(finetune_model_MNv3.history['accuracy'], label='accuracy')
# plt.plot(finetune_model_MNv3.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.ylim([0, 1])
# plt.legend(loc='lower right')

MNv3_test_loss, MNv3_test_acc = model_MNv3.evaluate(val_x,  val_y, verbose=2)

print(f"loss = {MNv3_test_loss}\nacc = {MNv3_test_acc}")
plt.show()

# np.random.shuffle(complete_set)

# split = 25

# train_x = np.array([i[0] for i in complete_set[:split]])
# train_y = np.array([i[1] for i in complete_set[:split]])
# val_x = np.array([i[0] for i in complete_set[split:]])
# val_y = np.array([i[1] for i in complete_set[split:]])

bm = model_MNv3.layers[1]
for layer in bm.layers[-min(len(bm.layers), 92):]:
 layer.trainable = True
model_MNv3.compile(optimizer=optimizers.Adam(learning_rate=0.000001),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
finetune_model_MNv3_1 = model_MNv3.fit(train_x, train_y, epochs=10,
                    validation_data=(val_x, val_y))
MNv3_test_loss, MNv3_test_acc = model_MNv3.evaluate(val_x,  val_y, verbose=2)
print(f'tune1 acc = {MNv3_test_acc}')

bm = model_MNv3.layers[1]
for layer in bm.layers[-min(len(bm.layers), 101):]:
 layer.trainable = True
model_MNv3.compile(optimizer=optimizers.Adam(learning_rate=0.000001/2),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
finetune_model_MNv3_2 = model_MNv3.fit(train_x, train_y, epochs=10,
                    validation_data=(val_x, val_y))
MNv3_test_loss, MNv3_test_acc = model_MNv3.evaluate(val_x,  val_y, verbose=2)
print(f'tune2 acc = {MNv3_test_acc}')

bm = model_MNv3.layers[1]
for layer in bm.layers[-min(len(bm.layers), 120):]:
 layer.trainable = True
model_MNv3.compile(optimizer=optimizers.Adam(learning_rate=0.000001/4),
             loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
             metrics=['accuracy'])
finetune_model_MNv3_3 = model_MNv3.fit(train_x, train_y, epochs=10,
                    validation_data=(val_x, val_y))
MNv3_test_loss, MNv3_test_acc = model_MNv3.evaluate(val_x,  val_y, verbose=2)
print(f'tune3 acc = {MNv3_test_acc}')


model_MNv3.save("Models/building_recognizer.keras")

plt.plot(finetune_model_MNv3_3.history['accuracy'], label='accuracy')
plt.plot(finetune_model_MNv3_3.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

class_names = ["normal", "broken"]

predict = model_MNv3.predict(tf.reshape(normal_set[5][0], (1, 192, 192, 3)))
score = tf.nn.softmax(predict[0])
print(predict)
print(score)
print(class_names[np.argmax(score)], 100 * np.max(score))

predict = model_MNv3.predict(tf.reshape(broken_set[2][0], (1, 192, 192, 3)))
score = tf.nn.softmax(predict[0])
print(predict)
print(score)
print(class_names[np.argmax(score)], 100 * np.max(score))