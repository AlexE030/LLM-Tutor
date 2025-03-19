import pdfplumber
import re
from chromadb.config import Settings
import chromadb

def is_heading(text):
    """
    Prüft, ob der Text als Überschrift gilt.
    Hier: Überschriften beginnen mit einer Zahlenfolge, z.B. "7." oder "7.1".
    """
    pattern = r'^\d+(?:\.\d+)*\s+'
    return re.match(pattern, text.strip()) is not None

def get_heading_level(text):
    """
    Bestimmt das Level der Überschrift anhand der Anzahl der Zahlenkomponenten.
    Beispiel: "7." → Level 1, "7.1" → Level 2, "7.1.3" → Level 3.
    """
    match = re.match(r'^(\d+(?:\.\d+)*)', text.strip())
    if match:
        segments = match.group(1).split('.')
        return len(segments)
    return None

def split_heading_from_line(line, max_heading_words=10):
    """
    Trennt, falls möglich, den Überschriftenanteil vom restlichen Fließtext.
    Wird angenommen, dass in einer Zeile mit Überschrift eventuell noch zusätzlicher Text steht.
    Gibt ein Tuple (heading, rest) zurück. Falls kein Rest existiert, ist rest leer.
    """
    if is_heading(line):
        words = line.strip().split()
        if len(words) > max_heading_words:
            heading_candidate = " ".join(words[:max_heading_words])
            if is_heading(heading_candidate):
                body = " ".join(words[max_heading_words:])
                return heading_candidate, body
        return line.strip(), ""
    return None, line.strip()

def extract_paragraphs_from_page(page, max_heading_words=10):
    """
    Extrahiert den Text einer Seite und fasst Zeilen zu Absätzen zusammen,
    bis ein Absatz durch ein Satzzeichen (".", "!" oder "?") abgeschlossen wird.
    Seitenzahlen im Format "Zahl / Zahl" werden übersprungen.
    Überschriften werden mithilfe von split_heading_from_line erkannt und separat ausgegeben.
    """
    text = page.extract_text()
    if not text:
        return []
    lines = text.split("\n")
    paragraphs = []
    current_paragraph = ""
    for line in lines:
        line = line.strip()
        # Überspringe Seitenzahlen im Format "15 / 55"
        if re.match(r'^\d+\s*/\s*\d+$', line):
            continue
        if not line:
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = ""
            continue
        # Prüfe, ob die Zeile als Überschrift erkannt wird.
        if is_heading(line):
            # Vorherigen Absatz abschließen, falls vorhanden.
            if current_paragraph:
                paragraphs.append(current_paragraph)
                current_paragraph = ""
            heading, rest = split_heading_from_line(line, max_heading_words=max_heading_words)
            paragraphs.append(heading)
            if rest:
                # Den Rest als neuen Absatz beginnen.
                current_paragraph = rest
            continue
        # Normalerweise: Zeile zum aktuellen Absatz hinzufügen.
        if current_paragraph:
            current_paragraph += " " + line
        else:
            current_paragraph = line
        # Falls der aktuelle Absatz mit einem Satzzeichen endet, abschließen.
        if current_paragraph and current_paragraph[-1] in ".!?":
            paragraphs.append(current_paragraph)
            current_paragraph = ""
    if current_paragraph:
        paragraphs.append(current_paragraph)
    return paragraphs

def extract_paragraph_chunks_with_headings(pdf_path, max_heading_words=10):
    """
    Öffnet die PDF und extrahiert seitenweise alle Absätze mithilfe der Satzzeichen-Heuristik.
    Währenddessen wird ein Heading-Context gepflegt:
      - Wird ein Absatz als Überschrift erkannt, wird der Kontext (alle übergeordneten Überschriften)
        aktualisiert.
      - Jeder Nicht-Überschrift-Absatz erhält den aktuellen Heading-Context als Metadatum.
    Die Seitenzahlen werden hier bewusst nicht gespeichert.
    """
    chunks = []
    heading_context = []  # z.B. ["7 Modul Bachelorarbeit …", "7.1 Anforderungen und Ablauf"]
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            paragraphs = extract_paragraphs_from_page(page, max_heading_words=max_heading_words)
            for paragraph in paragraphs:
                if is_heading(paragraph):
                    level = get_heading_level(paragraph)
                    if level is not None:
                        # Sorge dafür, dass der Context mindestens "level" Einträge enthält.
                        if len(heading_context) < level:
                            heading_context.extend([None] * (level - len(heading_context)))
                        # Setze auf der aktuellen Ebene die Überschrift und entferne alle tieferen Ebenen.
                        heading_context[level - 1] = paragraph
                        heading_context = heading_context[:level]
                    chunks.append({
                        "text": paragraph,
                        "headings": heading_context.copy()
                    })
                else:
                    chunks.append({
                        "text": paragraph,
                        "headings": heading_context.copy()
                    })
    return chunks

def assign_tags(text):
    """
    Weist themenspezifische Tags zu – hier beispielhaft für "Gliederung" und "Zitierweise".
    """
    tags = []
    gliederung_keywords = [
        "Gliederung", "Vorspann", "Problemstellung", "Hauptteil", "Kritische Reflexion", "Inhaltsverzeichnis"
    ]
    zitierweise_keywords = [
        "Zitierweise", "Quellennachweis", "Literaturverzeichnis", "APA", "MLA", "Chicago"
    ]
    for kw in gliederung_keywords:
        if kw.lower() in text.lower():
            tags.append("Gliederung")
            break
    for kw in zitierweise_keywords:
        if kw.lower() in text.lower():
            tags.append("Zitierweise")
            break
    return tags

# Pfad zur PDF (anpassen, falls nötig)
pdf_path = "./data/191212_Leitlinien_Praxismodule_Studien_Bachelorarbeiten.pdf"

# Extrahiere Absatz-Chunks inklusive vollständigem Heading-Context (ohne Seitenzahlen)
chunks = extract_paragraph_chunks_with_headings(pdf_path, max_heading_words=10)

# Ergänze themenspezifische Tags für jeden Chunk
for chunk in chunks:
    chunk["tags"] = assign_tags(chunk["text"])
    print(chunk["text"], chunk["headings"], chunk["tags"])

# Verbindung zu Chroma‑DB herstellen und Collection anlegen
# client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chroma_db"))
# collection = client.get_or_create_collection("leitlinien")
#
# # Füge alle Chunks in die Collection ein. Die Metadaten enthalten nur "headings" und "tags".
# for idx, chunk in enumerate(chunks):
#     doc_id = f"leitlinien_{idx}"
#     metadata = {
#         "headings": chunk["headings"],
#         "tags": chunk["tags"]
#     }
#     collection.add(
#         documents=[chunk["text"]],
#         metadatas=metadata,
#         ids=[doc_id]
#     )

print("Absatz-Chunks (getrennt anhand von Satzzeichen, ohne Seitenzahlen) mit vollständigem Heading-Context und Tags wurden in die Chroma‑DB eingefügt.")
