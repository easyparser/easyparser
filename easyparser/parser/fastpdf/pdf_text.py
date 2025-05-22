import pypdfium2


def get_text_pdfium(pdf_path: str):
    pdf = pypdfium2.PdfDocument(pdf_path)
    output = ""
    for idx in range(len(pdf)):
        page = pdf.get_page(idx)
        textpage = page.get_textpage()
        text = textpage.get_text_range()
        output += f"# Page {idx+1}\n" + text

    pdf.close()
    return output
