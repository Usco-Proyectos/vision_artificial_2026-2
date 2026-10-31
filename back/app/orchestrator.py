"""
orchestrator.py — Cascada lógica de seguridad (lógica PURA, sin torch/cv2).
Es el MISMO código que usa el servidor y el script de calibración.

M1 devuelve [p_ojo_der, p_ojo_izq]; basta con un ojo cerrado (OR).

Prioridad:
  C1  M1: ojos cerrados >= UMBRAL_FRAMES_OJOS frames seguidos      → 1 (regla fija, ignora el resto)
  C2  votación de los 4 modelos: alerta si al menos VOTOS_MINIMOS votan somnolencia → 1
        M1  hubo ojos cerrados sostenidos en la última ventana (~3,3 s)
        M2  hubo bostezo sostenido en la última ventana
        M3  la última ventana dice somnolencia
        M4  la última ventana dice somnolencia (Random Forest)
  BASE                                                             → 0
Con 2 votos de 4, en las ventanas de prueba de UTA-RLDD la exactitud fue 0,614 contra 0,633 de
la cascada anterior (C1 > M3 > bostezo), pero las falsas alarmas bajaron de 24,7 % a 17,3 %
(analisis/evaluar_ensemble.py del artículo). Un bostezo o M4 solos ya no dan alerta.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import config as C

VENTANA_S = C.VENTANA_M3 / C.FPS_M3  # el voto de M1 y M2 dura lo mismo que la ventana de M3/M4


@dataclass(frozen=True)
class Umbrales:
    prob_m1: float = C.UMBRAL_PROB_M1
    prob_m2: float = C.UMBRAL_PROB_M2
    frames_ojos: int = C.UMBRAL_FRAMES_OJOS
    frames_bostezo: int = C.UMBRAL_FRAMES_BOSTEZO
    votos_minimos: int = C.VOTOS_MINIMOS


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
    votos: List[str] = field(default_factory=list)  # modelos que votan somnolencia, p. ej. ["M3", "M4"]


class CascadaSeguridad:
    def __init__(self, umbrales: Umbrales = Umbrales()):
        self.u = umbrales
        self.reiniciar()

    def reiniciar(self) -> None:
        self._ojos = 0      # frames consecutivos con al menos un ojo cerrado (M1)
        self._bost = 0      # frames consecutivos con bostezo (M2)
        self.m3 = 0         # último veredicto de ventana (se mantiene hasta la siguiente)
        self.m4 = 0
        self._t_ojos = self._t_bost = float("-inf")  # última vez que M1 / M2 estuvieron sostenidos

    def step(self, m1_prob: Optional[Sequence[float]], m2_prob: Optional[float],
             m3_pred: Optional[int] = None, m4_pred: Optional[int] = None,
             t: Optional[float] = None) -> Veredicto:
        """Se llama UNA vez por frame de entrada. t en segundos (por defecto el reloj; las pruebas lo pasan)."""
        u = self.u
        t = time.monotonic() if t is None else t

        cerrado = m1_prob is not None and (m1_prob[0] >= u.prob_m1 or m1_prob[1] >= u.prob_m1)
        self._ojos = self._ojos + 1 if cerrado else 0

        yawn = m2_prob is not None and m2_prob >= u.prob_m2
        self._bost = self._bost + 1 if yawn else 0

        if m3_pred is not None:
            self.m3 = int(m3_pred)
        if m4_pred is not None:
            self.m4 = int(m4_pred)

        sostenido = self._ojos >= u.frames_ojos
        bostezo = self._bost >= u.frames_bostezo
        if sostenido:
            self._t_ojos = t
        if bostezo:
            self._t_bost = t

        votos = [m for m, si in (("M1", t - self._t_ojos <= VENTANA_S), ("M2", t - self._t_bost <= VENTANA_S),
                                 ("M3", self.m3 == 1), ("M4", self.m4 == 1)) if si]

        if sostenido:
            v, regla = 1, "C1_OJOS_CERRADOS_M1"
        elif len(votos) >= u.votos_minimos:
            v, regla = 1, "C2_VOTACION_" + "_".join(votos)
        else:
            v, regla = 0, "BASE"

        return Veredicto(v, regla, cerrado, self._ojos, sostenido, bostezo, self.m3, self.m4, votos)
