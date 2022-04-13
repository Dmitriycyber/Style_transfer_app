import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


class StyleAndContentExtractor:
    def __init__(self, style_layers, content_layers):
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.vgg_outputs_model = None
        self.num_style_layers = len(style_layers)
        self.vgg_outputs_model = get_vgg_layers_model(style_layers + content_layers)
        self.vgg_outputs_model.trainable = False

    def __call__(self, inputs):
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs * 255.)
        outputs = self.vgg_outputs_model(preprocessed_input)

        style_outputs_layers = outputs[:self.num_style_layers]
        content_outputs_layers = outputs[self.num_style_layers:]

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs_layers]
        content_outputs = [content_output for content_output in content_outputs_layers]

        features_dict = {}
        features_dict["style"] = {name: value for name, value in zip(self.style_layers, style_outputs)}
        features_dict["content"] = {name: value for name, value in zip(self.content_layers, content_outputs)}

        return features_dict


def style_content_loss(image, style_targets, content_targets, style_weight, content_weight, tv_weight, extractor):
    style_loss = None
    content_loss = None
    features_style = extractor(image)['style']
    features_content = extractor(image)['content']
    content_loss = tf.add_n([tf.keras.losses.MeanSquaredError()(features_content[name], content_targets[name])
                             for name in features_content.keys()])
    style_loss = tf.add_n([tf.keras.losses.MeanSquaredError()(features_style[name], style_targets[name])
                           for name in features_style.keys()])
    loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tf.image.total_variation(image)
    return loss


def train_step(image, loss_func, optimizer, style_targets, content_targets, style_weight, content_weight, tv_weight, extractor):
    with tf.GradientTape() as tape:
        loss = loss_func(image, style_targets, content_targets, style_weight, content_weight, tv_weight, extractor)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss.numpy()

def loss(image, style_targets, content_targets, style_weight, content_weight, tv_weight, extractor):
    return style_content_loss(image, style_targets, content_targets, style_weight, content_weight, tv_weight, extractor)

def clip_0_1(image):
    """
    Мы хотим уметь отображать нашу полученную картинку, а для этого ее значения должны
    находится в промежутке от 0 до 1. Наш алгоритм оптимизации этого нигде не учитывает
    поэтому к полученному изображению мы будем применять "обрезку" по значению

    """
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)


def load_img(path_to_img, max_dim=512):
    """
    Данная функция считывает изображение с диска и приводит его к такому размеру,
    чтобы бОльшая сторона была равна max_dim пикселей.

    Для считывания изображения воспользуемся функциями tensorflow.
    """
    img = tf.io.read_file(path_to_img)  # считываени файла
    img = tf.image.decode_image(img, channels=3)  # декодинг
    img = tf.image.convert_image_dtype(img, tf.float32)  # uint8 -> float32, 255 -> 1

    shape = img.numpy().shape[:-1]
    long_dim = max(shape)
    scale = max_dim / long_dim
    new_shape = tuple((np.array(shape) * scale).astype(np.int32))

    img = tf.image.resize(img, new_shape)  # изменение размера
    img = img[tf.newaxis, :]  # добавляем batch dimension
    return img


def imshow(image, title=None):
    """
    Функция для отрисовки изображения
    """
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.axis('off')
    plt.imshow(image)
    if title:
        plt.title(title)


def show_pair(original, generated, title=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    imshow(original, 'Original Image')
    plt.subplot(1, 2, 2)
    imshow(generated, title)


def get_vgg_layers_model(layer_names):
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(input_tensor):
    """
    В вычислении грам матрицы есть небольшие изменения по сравнению с прошлым уроком
    В этот раз мы делим не на W*H, а на W*H*C.
    Принципиально это ничего не меняет. Это один из способов по разному взвесить
    вклад разных слоев.
    Такое вычисление используется, например, здесь: https://arxiv.org/pdf/1603.08155.pdf
    """
    result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
    input_shape = tf.shape(input_tensor)

    num_locations = tf.cast(input_shape[1] * input_shape[2] * input_shape[3], tf.float32)
    return result / (num_locations)