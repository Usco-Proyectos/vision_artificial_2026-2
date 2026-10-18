"""Inserta secciones .md en un .docx ya armado, justo antes del título "Referencias".

Uso (desde la carpeta del proyecto):
  python src/compilar_entrega.py BASE.docx SALIDA.docx sec1.md sec2.md ...

Copia el formato del Hito 1: Times, tablas con borde arriba/abajo, rótulos en estilo
Descripción, ecuaciones centradas y x_w / x^{..} como subíndice / superíndice.
"""
import re
import sys
from pathlib import Path

import docx
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_COLOR_INDEX
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image

ANCHO_TABLA = 9360  # dxa, igual que la Tabla 1
LISTA_VINETA = 3     # numId de las viñetas del Hito 1
ABSTRACTO_NUMERADO = 2


def subindices(texto):
    """Parte 'x_{ab}^c' en trozos (texto, 'sub'|'sup'|None)."""
    trozos, buf, i = [], "", 0
    while i < len(texto):
        ch = texto[i]
        if ch in "_^" and i + 1 < len(texto) and (texto[i + 1] == "{" or texto[i + 1].isalnum()):
            if texto[i + 1] == "{":
                nivel, j = 0, i + 1
                while j < len(texto):
                    nivel += {"{": 1, "}": -1}.get(texto[j], 0)
                    if nivel == 0:
                        break
                    j += 1
                contenido, i = texto[i + 2:j], j + 1
            else:
                j = i + 1
                while j < len(texto) and (texto[j].isalnum() or texto[j] == "′"):
                    j += 1
                contenido, i = texto[i + 1:j], j
            trozos.append((buf, None))
            buf = ""
            trozos.append((re.sub(r"[_^{}]", "", contenido), "sub" if ch == "_" else "sup"))
            continue
        buf += ch
        i += 1
    trozos.append((buf, None))
    return [t for t in trozos if t[0]]


def escribir(parrafo, texto, tam=None, negrita_todo=False):
    for parte in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*|\[(?:REF FALTA|FIGURA PENDIENTE)[^\]]*\])", texto):
        if not parte:
            continue
        negrita = negrita_todo or parte.startswith("**")
        cursiva = not negrita and parte.startswith("*") and len(parte) > 2
        pendiente = parte.startswith("[REF FALTA") or parte.startswith("[FIGURA PENDIENTE")
        limpio = parte.strip("*") if (negrita and parte.startswith("**")) or cursiva else parte
        for trozo, pos in subindices(limpio):
            r = parrafo.add_run(trozo)
            r.bold, r.italic = negrita or None, cursiva or None
            if pos == "sub":
                r.font.subscript = True
            elif pos == "sup":
                r.font.superscript = True
            if tam:
                r.font.size = Pt(tam)
            if pendiente:
                r.font.highlight_color = WD_COLOR_INDEX.YELLOW
                r.font.color.rgb = RGBColor(0xC0, 0, 0)


def sin_sangria(p, centrado=False):
    p.paragraph_format.first_line_indent = 0
    if centrado:
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER


def rotulo(doc, texto, antes=240, despues=120, siguiente=True):
    p = doc.add_paragraph(style=doc.styles["Caption"])
    p.paragraph_format.space_before, p.paragraph_format.space_after = Pt(antes / 20), Pt(despues / 20)
    p.paragraph_format.keep_with_next = siguiente
    p.alignment = WD_ALIGN_PARAGRAPH.LEFT
    m = re.match(r"(Fuente|Nota):(.*)", texto)
    if m:  # igual que la "Nota:" del Hito 1
        r = p.add_run(m.group(1) + ":")
        r.italic = True
        escribir(p, m.group(2))
    else:
        escribir(p, texto)
    return p


def borde(celda, lados):
    tcPr = celda._tc.get_or_add_tcPr()
    bordes = OxmlElement("w:tcBorders")
    for lado in lados:
        b = OxmlElement(f"w:{lado}")
        for k, v in {"val": "single", "sz": "12", "space": "0", "color": "000000"}.items():
            b.set(qn(f"w:{k}"), v)
        bordes.append(b)
    tcPr.append(bordes)


def tabla(doc, filas):
    n = len(filas[0])
    t = doc.add_table(rows=len(filas), cols=n)
    t.autofit = False
    tblPr = t._tbl.tblPr
    ancho = OxmlElement("w:tblW")
    ancho.set(qn("w:w"), str(ANCHO_TABLA))
    ancho.set(qn("w:type"), "dxa")
    tblPr.append(ancho)
    for i, fila in enumerate(filas):
        if i == 0:
            trPr = t.rows[0]._tr.get_or_add_trPr()
            trPr.append(OxmlElement("w:tblHeader"))
        for j, texto in enumerate(fila):
            c = t.cell(i, j)
            c.width = docx.shared.Twips(ANCHO_TABLA // n)
            p = c.paragraphs[0]
            p.paragraph_format.space_after = 0
            p.paragraph_format.line_spacing = 1.0
            sin_sangria(p, centrado=i == 0)
            if i > 0:
                p.alignment = WD_ALIGN_PARAGRAPH.LEFT
            escribir(p, texto, tam=10, negrita_todo=i == 0)
            lados = (["top", "bottom"] if i == 0 else []) + (["bottom"] if i == len(filas) - 1 else [])
            if lados:
                borde(c, lados)
    return t


def lista_numerada_nueva(doc):
    numbering = doc.part.numbering_part.element
    ids = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]
    num = OxmlElement("w:num")
    num.set(qn("w:numId"), str(max(ids) + 1))
    abst = OxmlElement("w:abstractNumId")
    abst.set(qn("w:val"), str(ABSTRACTO_NUMERADO))
    num.append(abst)
    over = OxmlElement("w:lvlOverride")
    over.set(qn("w:ilvl"), "0")
    inicio = OxmlElement("w:startOverride")
    inicio.set(qn("w:val"), "1")
    over.append(inicio)
    num.append(over)
    numbering.append(num)
    return max(ids) + 1


def item(doc, texto, num_id):
    p = doc.add_paragraph(style=doc.styles["List Paragraph"])
    pPr = p._p.get_or_add_pPr()
    numPr = OxmlElement("w:numPr")
    for etiqueta, val in (("w:ilvl", "0"), ("w:numId", str(num_id))):
        e = OxmlElement(etiqueta)
        e.set(qn("w:val"), val)
        numPr.append(e)
    pPr.append(numPr)
    p.paragraph_format.left_indent, p.paragraph_format.first_line_indent = Inches(0.5), -Inches(436 / 1440)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    escribir(p, texto)


def agregar_md(doc, md, carpeta):
    md = re.sub(r"<!--.*?-->", "", md, flags=re.S)
    bloques = re.split(r"\n\s*\n", md.strip())
    for bloque in bloques:
        lineas = [l.rstrip() for l in bloque.split("\n") if l.strip()]
        primera = lineas[0]
        if primera.startswith("#"):
            nivel = len(primera) - len(primera.lstrip("#"))
            doc.add_heading(primera.lstrip("#").strip(), level=nivel)
        elif primera.startswith("|"):
            filas = [[c.strip() for c in l.strip("|").split("|")] for l in lineas
                     if not re.match(r"^\|[\s:|-]+\|$", l)]
            tabla(doc, filas)
        elif primera.startswith("!["):
            ruta = (carpeta / re.search(r"\]\(([^)]+)\)", primera).group(1)).resolve()
            w, h = Image.open(ruta).size
            ancho = min(6.0, 7.5 * w / h)  # que no pase de 7,5" de alto
            p = doc.add_paragraph()
            sin_sangria(p, centrado=True)
            p.paragraph_format.keep_with_next = True
            p.add_run().add_picture(str(ruta), width=Inches(ancho))
        elif re.match(r"\*\*(Tabla|Figura) \d+\.\*\*", primera):
            rotulo(doc, primera)
        elif re.match(r"(Fuente|Nota):", primera):
            rotulo(doc, primera, antes=60, despues=240, siguiente=False)
        elif re.match(r"(\d+\.|-) ", primera):
            num_id = lista_numerada_nueva(doc) if primera[0].isdigit() else LISTA_VINETA
            for l in lineas:
                item(doc, re.sub(r"^(\d+\.|-)\s+", "", l.strip()), num_id)
        elif re.search(r"\(\d+\)$", lineas[-1]):  # ecuación numerada, una línea por renglón
            for l in lineas:
                p = doc.add_paragraph()
                sin_sangria(p, centrado=True)
                p.paragraph_format.space_before = p.paragraph_format.space_after = Pt(6)
                escribir(p, l)
        else:
            escribir(doc.add_paragraph(), " ".join(lineas))


def main():
    base, salida, *secciones = sys.argv[1:]
    if Path(salida).exists():
        sys.exit(f"{salida} ya existe; no se sobrescriben entregas.")
    doc = docx.Document(base)
    cuerpo = doc.element.body
    refs = next(p._p for p in doc.paragraphs
                if p.style.name == "Heading 2" and p.text.strip() == "Referencias")
    n_antes = len(cuerpo) - 1  # python-docx agrega antes del sectPr final
    for s in secciones:
        agregar_md(doc, Path(s).read_text(encoding="utf-8"), Path(s).parent)
    nuevos = list(cuerpo)[n_antes:-1]
    assert cuerpo[-1].tag.endswith("sectPr")
    for e in nuevos:
        refs.addprevious(e)
    doc.save(salida)
    print(f"{len(nuevos)} elementos insertados -> {salida}")


if __name__ == "__main__":
    main()
