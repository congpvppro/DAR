from pathlib import Path
import re

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.shared import Inches, Pt, RGBColor
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SRC = HERE / 'report.vi.md'
OUT = HERE / 'report.vi.docx'
BLACK = RGBColor(0, 0, 0)

def set_run_black(run, bold=None, italic=None, size=None, font='Aptos'):
    run.font.name = font
    run._element.get_or_add_rPr().rFonts.set(qn('w:ascii'), font)
    run._element.get_or_add_rPr().rFonts.set(qn('w:hAnsi'), font)
    run.font.color.rgb = BLACK
    if bold is not None: run.bold = bold
    if italic is not None: run.italic = italic
    if size is not None: run.font.size = Pt(size)

def shade(cell, fill):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = tcPr.find(qn('w:shd')) or OxmlElement('w:shd')
    shd.set(qn('w:fill'), fill)
    if shd.getparent() is None: tcPr.append(shd)

def borders(table, color='D9D9D9', sz='6'):
    tblPr = table._tbl.tblPr
    b = tblPr.first_child_found_in('w:tblBorders')
    if b is None:
        b = OxmlElement('w:tblBorders'); tblPr.append(b)
    for edge in ('top','left','bottom','right','insideH','insideV'):
        tag = 'w:' + edge
        el = b.find(qn(tag)) or OxmlElement(tag)
        el.set(qn('w:val'), 'single'); el.set(qn('w:sz'), sz); el.set(qn('w:space'), '0'); el.set(qn('w:color'), color)
        if el.getparent() is None: b.append(el)

def set_cell_text(cell, text, bold=False, size=8.5):
    cell.text = ''
    p = cell.paragraphs[0]
    p.paragraph_format.space_after = Pt(2)
    p.paragraph_format.space_before = Pt(1)
    p.paragraph_format.line_spacing = 1.05
    r = p.add_run(clean_inline(text))
    set_run_black(r, bold=bold, size=size)
    cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER

def clean_inline(text):
    text = re.sub(r'!\[([^]]*)\]\([^)]*\)', r'\1', text)
    text = re.sub(r'\[([^]]+)\]\(([^)]+)\)', r'\1', text)
    text = re.sub(r'`([^`]*)`', r'\1', text)
    text = text.replace('**','').replace('__','')
    text = text.replace('\\*','*')
    return text

def add_rich_paragraph(doc, text, style=None, size=10.2, align=None):
    p = doc.add_paragraph(style=style)
    if align is not None: p.alignment = align
    p.paragraph_format.space_after = Pt(6)
    p.paragraph_format.line_spacing = 1.12
    # Keep readable emphasis while ensuring every run is black.
    pattern = re.compile(r'(\*\*.*?\*\*|`.*?`|\*.*?\*)')
    pos = 0
    for m in pattern.finditer(text):
        if m.start() > pos:
            set_run_black(p.add_run(clean_inline(text[pos:m.start()])), size=size)
        token = m.group(0)
        if token.startswith('**'):
            set_run_black(p.add_run(clean_inline(token[2:-2])), bold=True, size=size)
        elif token.startswith('`'):
            set_run_black(p.add_run(clean_inline(token[1:-1])), font='Consolas', size=size-0.5)
        else:
            set_run_black(p.add_run(clean_inline(token[1:-1])), italic=True, size=size)
        pos = m.end()
    if pos < len(text): set_run_black(p.add_run(clean_inline(text[pos:])), size=size)
    return p

def add_table(doc, lines):
    rows = []
    for line in lines:
        if not line.strip().startswith('|'): continue
        parts = [x.strip() for x in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-{3,}:?', x) for x in parts): continue
        rows.append(parts)
    if not rows: return
    cols = max(len(r) for r in rows)
    table = doc.add_table(rows=len(rows), cols=cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    borders(table)
    for i, row in enumerate(rows):
        for j in range(cols):
            txt = row[j] if j < len(row) else ''
            set_cell_text(table.cell(i,j), txt, bold=(i==0), size=8.1 if cols>=4 else 8.5)
            if i == 0: shade(table.cell(i,j), 'D9EAF2')
            elif i % 2 == 0: shade(table.cell(i,j), 'F7FAFC')
    doc.add_paragraph().paragraph_format.space_after = Pt(2)

def add_image(doc, alt, rel):
    path = HERE / rel
    if not path.exists(): return
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
    run = p.add_run(); run.add_picture(str(path), width=Inches(6.35))
    set_run_black(run)
    cap = doc.add_paragraph()
    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap.paragraph_format.space_after = Pt(8)
    set_run_black(cap.add_run(alt), italic=True, size=9)

def build():
    doc = Document()
    sec = doc.sections[0]
    sec.top_margin = Inches(.72); sec.bottom_margin = Inches(.7)
    sec.left_margin = Inches(.8); sec.right_margin = Inches(.8)
    styles = doc.styles
    styles['Normal'].font.name = 'Aptos'; styles['Normal']._element.rPr.rFonts.set(qn('w:ascii'),'Aptos'); styles['Normal']._element.rPr.rFonts.set(qn('w:hAnsi'),'Aptos'); styles['Normal'].font.color.rgb = BLACK; styles['Normal'].font.size = Pt(10.2)
    for name, size in [('Title',20),('Heading 1',15),('Heading 2',12.5),('Heading 3',11)]:
        st = styles[name]; st.font.name='Aptos Display' if name=='Title' else 'Aptos'; st._element.rPr.rFonts.set(qn('w:ascii'), st.font.name); st._element.rPr.rFonts.set(qn('w:hAnsi'), st.font.name); st.font.color.rgb=BLACK; st.font.size=Pt(size); st.font.bold=True
    # footer with page number field
    footer = sec.footer.paragraphs[0]; footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = footer.add_run('DAR survey report  •  '); set_run_black(r, size=8)
    fld = OxmlElement('w:fldSimple'); fld.set(qn('w:instr'), 'PAGE'); footer._p.append(fld)
    lines = SRC.read_text(encoding='utf-8').splitlines()
    i = 0; first = True
    while i < len(lines):
        line = lines[i]
        if not line.strip(): i += 1; continue
        m = re.match(r'^(#{1,3})\s+(.*)$', line)
        if m:
            level = len(m.group(1)); txt = clean_inline(m.group(2))
            style = 'Title' if level==1 and first else ('Heading 1' if level==2 else 'Heading 2')
            p = doc.add_paragraph(style=style); p.paragraph_format.space_before = Pt(14 if level>1 else 0); p.paragraph_format.space_after = Pt(7)
            r=p.add_run(txt); set_run_black(r, bold=True, size=20 if style=='Title' else (15 if style=='Heading 1' else 12.5))
            first = False; i += 1; continue
        if line.startswith('|'):
            group=[]
            while i < len(lines) and lines[i].strip().startswith('|'):
                group.append(lines[i]); i += 1
            add_table(doc, group); continue
        img = re.match(r'^!\[([^]]*)\]\(([^)]+)\)', line)
        if img:
            add_image(doc, img.group(1), img.group(2)); i += 1; continue
        if line.startswith('```'):
            code=[]; i += 1
            while i < len(lines) and not lines[i].startswith('```'):
                code.append(lines[i]); i += 1
            i += 1
            p=doc.add_paragraph(); p.paragraph_format.left_indent=Inches(.25); p.paragraph_format.space_after=Pt(8)
            r=p.add_run('\n'.join(code)); set_run_black(r, size=8.3, font='Consolas'); continue
        # Join ordinary continuation lines into a paragraph; retain list markers as text.
        para=[line]; i += 1
        while i < len(lines) and lines[i].strip() and not re.match(r'^(#{1,3})\s+', lines[i]) and not lines[i].startswith('|') and not lines[i].startswith('![') and not lines[i].startswith('```'):
            para.append(lines[i]); i += 1
        add_rich_paragraph(doc, ' '.join(x.strip() for x in para))
    # Ensure all existing runs are black, including fields/late-added content.
    for p in doc.paragraphs:
        for r in p.runs: r.font.color.rgb = BLACK
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                for p in c.paragraphs:
                    for r in p.runs: r.font.color.rgb = BLACK
    doc.core_properties.title = 'Bốn survey về multimodal reasoning và cải tiến DAR'
    doc.core_properties.subject = 'Báo cáo nghiên cứu đối chiếu survey với DAR'
    doc.save(OUT)
    print(OUT)

if __name__ == '__main__': build()
