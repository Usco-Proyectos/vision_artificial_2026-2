"""
models_loader.py — Carga de los 4 modelos (.pt con torch, .joblib con joblib).
CORREGIDO para los modelos reales del proyecto:
  - M1/M2: MobileNetV3-Small con state_dict, cabeza classifier[3] de 1 logit
  - M3: CnnLstm(senales=True) con buffers media/desv de tamaño 6
  - M4: Pipeline SVM o RandomForest de scikit-learn
"""
import logging
from dataclasses import dataclass
from typing import Any, Callable

import joblib
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small

from . import config as C

log = logging.getLogger("modelos")
DEVICE = torch.device("cuda" if torch.cuda.is_available() and C.DEVICE_PREFERIDO == "cuda" else "cpu")


# ══════════════════════════════════════════════════════════════════════════════
# CONSTRUCTORES DE MODELOS (deben coincidir EXACTAMENTE con el entrenamiento)
# ══════════════════════════════════════════════════════════════════════════════

def construir_mobilenet_binario() -> nn.Module:
    """
    M1 / M2: MobileNetV3-Small con cabeza de 1 logit.
    CAMBIO CRÍTICO: usar classifier[3] en vez de classifier[-1] para que
    las claves del state_dict coincidan (classifier.3.weight, classifier.3.bias).
    """
    m = mobilenet_v3_small(weights=None)
    # CAMBIO: índice [3] en vez de [-1]
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 1)
    return m


class CnnLstm(nn.Module):
    """
    M3: LSTM sobre señales (Opción B confirmada).
    Esta es la clase EXACTA de entrenar_m3.py. NO usa CNN.
    Entrada: (B, T, 6) donde T=10 y las 6 señales son:
        [p_ojo_der, p_ojo_izq, p_boca, ear_der, ear_izq, mar]
    Los frames sin rostro se rellenan con NaN al final del tensor.
    """
    def __init__(self, senales: bool = True, oculto: int = 128):
        super().__init__()
        self.senales = senales
        if senales:
            self.cnn = None
            n_senales = len(C.SENALES_M3)
            self.register_buffer("media", torch.zeros(n_senales))
            self.register_buffer("desv", torch.ones(n_senales))
            self.lstm = nn.LSTM(n_senales, oculto, batch_first=True)
        else:
            # Este branch NO se usa (tu M3 es de señales), pero lo dejamos por completitud
            self.cnn = construir_mobilenet_binario()
            self.cnn.classifier[3] = nn.Identity()  # features de 1024 dims
            self.lstm = nn.LSTM(1024, oculto, batch_first=True)
        
        self.salida = nn.Linear(oculto, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor de forma (B, T, 6) con NaN al final para frames inválidos
        Returns:
            Logits de forma (B, 1)
        """
        b = x.shape[0]
        # Detectar frames reales (no NaN) mirando la primera característica
        reales = ~torch.isnan(x[:, :, 0])  # (B, T) boolean mask
        
        if self.cnn is None:
            # Modo señales: estandarizar con los buffers guardados
            # Solo procesar frames reales para evitar propagar NaN
            f = torch.zeros(b, x.shape[1], x.shape[2], device=x.device, dtype=x.dtype)
            for i in range(b):
                mask_i = reales[i]
                if mask_i.any():
                    f[i, mask_i] = (x[i, mask_i] - self.media) / self.desv
        else:
            # Modo imágenes (no usado en tu caso)
            f = self.cnn(x[reales])
        
        # Empaquetar secuencia para LSTM (ignora los NaN del final)
        from torch.nn.utils.rnn import pack_padded_sequence
        longitudes = reales.sum(1).cpu()
        sec = pack_padded_sequence(f, longitudes, batch_first=True, enforce_sorted=False)
        _, (h, _) = self.lstm(sec)
        return self.salida(h[-1])  # (B, 1) logit


# ══════════════════════════════════════════════════════════════════════════════
# FUNCIONES DE CARGA
# ══════════════════════════════════════════════════════════════════════════════

def _cargar_state_dict(ruta: str, constructor: Callable[[], nn.Module]) -> nn.Module:
    """
    Carga un state_dict de PyTorch en un modelo construido.
    Simplificado: asumimos que tus .pt son state_dicts puros (no checkpoints).
    """
    try:
        # Cargar el state_dict directamente
        state_dict = torch.load(str(ruta), map_location=DEVICE, weights_only=True)
        
        # Construir el modelo vacío
        modelo = constructor()
        
        # Cargar los pesos
        modelo.load_state_dict(state_dict, strict=True)
        
        log.info(" Cargado %s en %s (state_dict)", ruta.name, DEVICE)
        return modelo.to(DEVICE).eval()
        
    except Exception as e:
        log.error(" Error al cargar %s: %s", ruta.name, str(e))
        raise


def a_prob(salida: torch.Tensor) -> torch.Tensor:
    """
    Convierte la salida del modelo en P(clase positiva).
    - 1 logit → sigmoide
    - 2 logits → softmax[:, 1]
    """
    if salida.shape[-1] == 1:
        return torch.sigmoid(salida).flatten()
    return torch.softmax(salida, dim=-1)[..., 1]


def _validar_modelo(modelo: nn.Module, nombre: str) -> None:
    """Verifica que el modelo tenga los parámetros esperados."""
    params = sum(p.numel() for p in modelo.parameters())
    log.info("   %s: %d parámetros", nombre, params)
    
    # Verificación específica para M3
    if nombre == "m3" and hasattr(modelo, 'media'):
        log.info("   M3 media shape: %s, desv shape: %s", 
                tuple(modelo.media.shape), tuple(modelo.desv.shape))


# ══════════════════════════════════════════════════════════════════════════════
# ESTRUCTURA DE DATOS Y FUNCIÓN PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Modelos:
    m1: nn.Module      # MobileNetV3-Small para ojos (96x96 gris→3 canales)
    m2: nn.Module      # MobileNetV3-Small para boca (96x96 gris→3 canales)
    m3: nn.Module      # CnnLstm(senales=True) para somnolencia por video
    m4: Any            # Pipeline SVM o RandomForest de scikit-learn


def cargar_modelos() -> Modelos:
    """
    Carga los 4 modelos en memoria y los devuelve listos para inferencia.
    """
    log.info("=" * 60)
    log.info("Cargando modelos...")
    log.info("Dispositivo: %s", DEVICE)
    
    # Configuración de PyTorch
    torch.set_num_threads(C.NUM_HILOS_TORCH)
    if DEVICE.type == "cuda":
        torch.backends.cudnn.benchmark = True
        log.info("CUDA activado con cudnn.benchmark")
    
    # Cargar modelos PyTorch
    log.info("\n Cargando M1 (ojos)...")
    m1 = _cargar_state_dict(C.RUTA_M1, construir_mobilenet_binario)
    _validar_modelo(m1, "m1")
    
    log.info("\n Cargando M2 (boca)...")
    m2 = _cargar_state_dict(C.RUTA_M2, construir_mobilenet_binario)
    _validar_modelo(m2, "m2")
    
    log.info("\n Cargando M3 (somnolencia LSTM)...")
    m3 = _cargar_state_dict(C.RUTA_M3, lambda: CnnLstm(senales=True, oculto=128))
    _validar_modelo(m3, "m3")
    
    # Cargar modelo scikit-learn
    log.info("\n Cargando M4 (descriptores)...")
    m4 = joblib.load(C.RUTA_M4)
    log.info(" Cargado %s", C.RUTA_M4.name)
    
    # Validar que M4 tenga las features esperadas
    if hasattr(m4, 'n_features_in_'):
        log.info("   M4 espera %d features", m4.n_features_in_)
        if m4.n_features_in_ != len(C.FEATURES_M4):
            log.warning("  M4 espera %d features pero config.py define %d",
                       m4.n_features_in_, len(C.FEATURES_M4))
    
    log.info("\n" + "=" * 60)
    log.info(" Todos los modelos cargados exitosamente")
    log.info("=" * 60)
    
    return Modelos(m1=m1, m2=m2, m3=m3, m4=m4)