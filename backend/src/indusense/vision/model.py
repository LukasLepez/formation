"""Architecture Keras 3 de l'auto-encodeur convolutionnel du TP B6."""

from __future__ import annotations

import os
from contextlib import redirect_stdout
from io import StringIO
from typing import Any


def keras_api() -> Any:
    """Charge Keras tardivement avec un backend compatible avec Python 3.14."""

    os.environ.setdefault("KERAS_BACKEND", "torch")
    try:
        import keras
    except ImportError as error:
        raise RuntimeError(
            "La partie vision requiert Keras 3 et PyTorch. Lancez `uv sync` dans backend."
        ) from error
    return keras


def ssim_loss(y_true: Any, y_pred: Any) -> Any:
    """Perte SSIM différentiable, calculée par image et par canal.

    Cette variante globale et multicanal conserve les trois composantes de la
    SSIM (luminance, contraste et structure) et reste portable entre les
    backends Keras 3.
    """

    keras = keras_api()
    ops = keras.ops
    axes = (1, 2)
    mean_true = ops.mean(y_true, axis=axes, keepdims=True)
    mean_pred = ops.mean(y_pred, axis=axes, keepdims=True)
    centered_true = y_true - mean_true
    centered_pred = y_pred - mean_pred
    variance_true = ops.mean(ops.square(centered_true), axis=axes, keepdims=True)
    variance_pred = ops.mean(ops.square(centered_pred), axis=axes, keepdims=True)
    covariance = ops.mean(centered_true * centered_pred, axis=axes, keepdims=True)
    c1 = 0.01**2
    c2 = 0.03**2
    numerator = (2.0 * mean_true * mean_pred + c1) * (2.0 * covariance + c2)
    denominator = (ops.square(mean_true) + ops.square(mean_pred) + c1) * (
        variance_true + variance_pred + c2
    )
    similarity = numerator / (denominator + keras.backend.epsilon())
    return 1.0 - ops.mean(similarity)


def build_autoencoder(
    input_shape: tuple[int, int, int] = (256, 256, 3),
    *,
    latent_filters: int = 16,
    learning_rate: float = 1e-3,
    loss_name: str = "mse",
) -> Any:
    """Construit un auto-encodeur à 3 convolutions et 3 déconvolutions."""

    if input_shape[0] % 8 or input_shape[1] % 8:
        raise ValueError("La hauteur et la largeur doivent être divisibles par 8.")
    if latent_filters < 1:
        raise ValueError("Le nombre de filtres latents doit être positif.")
    if loss_name not in {"mse", "ssim"}:
        raise ValueError("La perte doit être `mse` ou `ssim`.")

    keras = keras_api()
    inputs = keras.Input(shape=input_shape, name="image")
    encoded = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu", name="encoder_conv_1")(inputs)
    encoded = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu", name="encoder_conv_2")(encoded)
    latent = keras.layers.Conv2D(
        latent_filters,
        3,
        strides=2,
        padding="same",
        activation="relu",
        name="latent",
    )(encoded)
    decoded = keras.layers.Conv2DTranspose(64, 3, strides=2, padding="same", activation="relu", name="decoder_deconv_1")(latent)
    decoded = keras.layers.Conv2DTranspose(32, 3, strides=2, padding="same", activation="relu", name="decoder_deconv_2")(decoded)
    outputs = keras.layers.Conv2DTranspose(
        input_shape[-1],
        3,
        strides=2,
        padding="same",
        activation="sigmoid",
        name="reconstruction",
    )(decoded)
    model = keras.Model(inputs, outputs, name="mvtec_bottle_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse" if loss_name == "mse" else ssim_loss,
    )
    return model


def model_summary_text(model: Any) -> str:
    output = StringIO()
    with redirect_stdout(output):
        model.summary()
    return output.getvalue()


def model_description(model: Any) -> dict[str, Any]:
    """Retourne les paramètres et le ratio de compression à commenter."""

    input_shape = tuple(int(value) for value in model.input_shape[1:])
    latent_shape = tuple(int(value) for value in model.get_layer("latent").output.shape[1:])
    input_values = _shape_product(input_shape)
    latent_values = _shape_product(latent_shape)
    ratio = input_values / latent_values
    input_values_text = f"{input_values:,}".replace(",", " ")
    latent_values_text = f"{latent_values:,}".replace(",", " ")
    ratio_text = f"{ratio:.2f}".replace(".", ",")
    return {
        "input_shape": list(input_shape),
        "latent_shape": list(latent_shape),
        "input_values": input_values,
        "latent_values": latent_values,
        "compression_ratio": round(ratio, 4),
        "parameter_count": int(model.count_params()),
        "comment": (
            f"Le latent contient {latent_values_text} valeurs contre {input_values_text} en entrée "
            f"(compression ×{ratio_text}). "
            + (
                "Le goulot est volontairement resserré pour limiter l'apprentissage d'une identité."
                if ratio >= 4
                else "Le ratio reste faible : le modèle risque de reconstruire aussi les défauts."
            )
        ),
    }


def _shape_product(shape: tuple[int, ...]) -> int:
    result = 1
    for value in shape:
        result *= value
    return result
