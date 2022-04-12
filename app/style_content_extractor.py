import tensorflow as tf
from app.functions import gram_matrix, clip_0_1
from app.functions import get_vgg_layers_model


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
        features_dict["app"] = {name: value for name, value in zip(self.content_layers, content_outputs)}

        return features_dict


def style_content_loss(image, style_targets, content_targets, style_weight, content_weight, tv_weight, extractor):
    style_loss = None
    content_loss = None
    features_style = extractor(image)['style']
    features_content = extractor(image)['content'],
    content_loss = tf.add_n([tf.keras.losses.MeanSquaredError()(features_content[name], content_targets[name])
                             for name in features_content.keys()])
    style_loss = tf.add_n([tf.keras.losses.MeanSquaredError()(features_style[name], style_targets[name])
                           for name in features_style.keys()])
    loss = style_weight * style_loss + content_weight * content_loss + tv_weight * tf.image.total_variation(image)
    return loss


def train_step(image, loss_func, optimizer):
    with tf.GradientTape() as tape:
        loss = loss_func(image)
    grad = tape.gradient(loss, image)
    optimizer.apply_gradients([(grad, image)])
    image.assign(clip_0_1(image))
    return loss.numpy()