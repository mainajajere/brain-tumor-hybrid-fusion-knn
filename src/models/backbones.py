import tensorflow as tf

def mobilenet_v2_backbone(image_size: int = 224, freeze: bool = True):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )
    base.trainable = not freeze
    return base

def efficientnet_v2_b0_backbone(image_size: int = 224, freeze: bool = True):
    base = tf.keras.applications.EfficientNetV2B0(
        include_top=False,
        weights="imagenet",
        input_shape=(image_size, image_size, 3)
    )
    base.trainable = not freeze
    return base

def build_dual_backbones(image_size: int = 224, freeze: bool = True):
    mb = mobilenet_v2_backbone(image_size, freeze)
    ef = efficientnet_v2_b0_backbone(image_size, freeze)
    return mb, ef
