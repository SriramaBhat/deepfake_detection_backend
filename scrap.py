import docx

def convert_docx_to_txt(docx_path):
    doc = docx.Document(docx_path)
    text = ''
    for paragraph in doc.paragraphs:
        text += paragraph.text + '\n'
    return text

# Example usage:
docx_path = 'deepfake_detection_backend\example.docx'
text = convert_docx_to_txt(docx_path)
print(text)
