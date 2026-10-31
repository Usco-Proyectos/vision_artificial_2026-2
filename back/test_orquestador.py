"""Prueba de la cascada (orchestrator.py) sin modelos ni cámara: python test_orquestador.py (desde back/)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from app.orchestrator import VENTANA_S, CascadaSeguridad

ABIERTOS, CERRADOS = [0.1, 0.1], [0.9, 0.1]  # basta con un ojo cerrado
t = 0.0


def regla(c, m1=ABIERTOS, m2=0.0, m3=None, m4=None, dt=0.1):
    global t
    t += dt
    return c.step(m1, m2, m3, m4, t=t).regla


c = CascadaSeguridad()
assert regla(c) == "BASE"

# un solo voto no alcanza: ni M4, ni M3, ni el bostezo solos
assert regla(c, m4=1) == "BASE"
assert regla(c, m4=0, m3=1) == "BASE"
assert regla(c, m3=0, m2=0.9) == "BASE"

# dos votos sí: bostezo reciente (sigue votando ~3,3 s) + M4
assert regla(c, m4=1) == "C2_VOTACION_M2_M4"
assert regla(c, m3=1) == "C2_VOTACION_M2_M3_M4"
# pasada la ventana el bostezo deja de votar; M3 y M4 se mantienen hasta la siguiente ventana
assert regla(c, dt=VENTANA_S + 0.1) == "C2_VOTACION_M3_M4"
assert regla(c, m3=0, m4=0) == "BASE"

# C1 gana sobre todo después de 3 fotogramas seguidos con un ojo cerrado
c = CascadaSeguridad()
assert regla(c, m1=CERRADOS) == "BASE"
assert regla(c, m1=CERRADOS) == "BASE"
assert regla(c, m1=CERRADOS) == "C1_OJOS_CERRADOS_M1"
assert regla(c, m1=None) == "BASE"  # sin rostro se corta la racha; M1 sigue votando pero solo
# ojos cerrados hace poco + M3 = 2 votos
assert regla(c, m3=1) == "C2_VOTACION_M1_M3"
assert regla(c, dt=VENTANA_S + 0.1) == "BASE"

print("cascada OK: C1 fija y luego al menos 2 de 4 votos (M1, M2, M3, M4)")
