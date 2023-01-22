import tensorflow as tf


class BuildModel:
    INP_SIZE = 224

    def __init__(self, name, is_fine_tuning=False) -> None:
        self.is_fine_tuning = is_fine_tuning
        self.basemodel = self.build_base(self.is_fine_tuning)
        self.model = self.build_base(self.is_fine_tuning)

    def build_seq(self):
        return tf.keras.Sequential([self.basemodel,
                                    tf.keras.layers.GlobalAveragePooling2D(),
                                    tf.keras.layers.Dense(101, activation='softmax'),
                                    ])

    def build_base(self, is_fine_tuning):
        return None


class MobilNat(BuildModel):
    def build_base(self, is_fine_tuning):
        model = tf.keras.applications.MobileNetV2(
            input_shape=(self.INP_SIZE, self.INP_SIZE, 3),
            include_top=False,
            weights='imagenet')
        model.trainable = is_fine_tuning
        return model


class EfficientNet(BuildModel):
    def build_base(self, is_fine_tuning):
        model = tf.keras.applications.EfficientNetV2S(
            input_shape=(self.INP_SIZE, self.INP_SIZE, 3),
            include_top=False,
            weights='imagenet',
            include_preprocessing=False, )
        model.trainable = is_fine_tuning
        return model
