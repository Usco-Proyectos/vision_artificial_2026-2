"""
orchestrator.py — Cascada lógica de seguridad (lógica PURA, sin torch/cv2).
Es el MISMO código que usa el servidor y el script de calibración.

CORREGIDO: M1 ahora devuelve [p_ojo_der, p_ojo_izq], se aplica OR lógico.

Prioridad (cortocircuito):
  C1  M1: ojos cerrados > UMBRAL_FRAMES_OJOS frames seguidos      → 1 (ignora el resto)
  C2  M3: la última ventana de 10 frames dice somnolencia          → 1
  C3  M2 (bostezo) Y M4 (descriptores) activos a la vez            → 1
  BASE                                                             → 0
"""
from dataclasses import dataclass
from typing import Optional, Sequence

from . import config as C


@dataclass(frozen=True)
class Umbrales:
    prob_m1: float = C.UMBRAL_PROB_M1
    prob_m2: float = C.UMBRAL_PROB_M2
    frames_ojos: int = C.UMBRAL_FRAMES_OJOS
    frames_bostezo: int = C.UMBRAL_FRAMES_BOSTEZO


@dataclass
class Veredicto:
    veredicto: int
    regla: str
    m1_cerrado_frame: bool      # True si AL MENOS UN ojo está cerrado
    m1_frames_seguidos: int
    m1_sostenido: bool
    m2_bostezo: bool
    m3: int
    m4: int


class CascadaSeguridad:
    def __init__(self, umbrales: Umbrales = Umbrales()):
        self.u = umbrales
        self.reiniciar()

    def reiniciar(self) -> None:
        self._ojos = 0      # frames consecutivos con al menos un ojo cerrado (M1)
        self._bost = 0      # frames consecutivos con bostezo (M2)
        self.m3 = 0         # último veredicto de ventana (se mantiene hasta la siguiente)
        self.m4 = 0

    def step(self, m1_prob: Optional[Sequence[float]], m2_prob: Optional[float],
             m3_pred: Optional[int] = None, m4_pred: Optional[int] = None) -> Veredicto:
        """
        Se llama UNA vez por frame de entrada (30 FPS).
        
        Args:
            m1_prob: lista [p_ojo_der, p_ojo_izq] o None si no hubo rostro
            m2_prob: float o None si no hubo rostro
            m3_pred: predicción de M3 (0 o 1) o None si no hay ventana nueva
            m4_pred: predicción de M4 (0 o 1) o None si no hay ventana nueva
        
        Returns:
            Veredicto con el resultado de la cascada
        """
        u = self.u
        
        # ── M1: ojo cerrado si AL MENOS UNO está cerrado (OR lógico) ──
        if m1_prob is not None:
            cerrado_der = m1_prob[0] >= u.prob_m1
            cerrado_izq = m1_prob[1] >= u.prob_m1
            cerrado = cerrado_der or cerrado_izq  # OR lógico
        else:
            cerrado = False
        
        self._ojos = self._ojos + 1 if cerrado else 0
        
        # ── M2: bostezo ──
        yawn = m2_prob is not None and m2_prob >= u.prob_m2
        self._bost = self._bost + 1 if yawn else 0

        # ── M3 y M4: actualizar si hay predicción nueva ──
        if m3_pred is not None:
            self.m3 = int(m3_pred)
        if m4_pred is not None:
            self.m4 = int(m4_pred)

        # ── Aplicar cascada (cortocircuito) ──
        sostenido = self._ojos > u.frames_ojos          # "por MÁS de N frames seguidos"
        bostezo = self._bost >= u.frames_bostezo

        if sostenido:
            v, regla = 1, "C1_OJOS_CERRADOS_M1"
        elif self.m3 == 1:
            v, regla = 1, "C2_SOMNOLENCIA_M3"
        elif bostezo and self.m4 == 1:
            v, regla = 1, "C3_BOSTEZO_M2_Y_DESCRIPTORES_M4"
        else:
            v, regla = 0, "BASE"
        
        return Veredicto(v, regla, cerrado, self._ojos, sostenido, bostezo, self.m3, self.m4)