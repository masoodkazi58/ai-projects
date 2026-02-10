from pathlib import Path
from pypdf import PdfReader


def extract_pdf_text_to_txt(pdf_path: str | Path, output_dir: str | Path) -> Path:
    """
    Extracts text from a PDF and saves it as a .txt file
    with the same name as the PDF.

    Args:
        pdf_path (str | Path): Full path to the PDF file
        output_dir (str | Path): Directory where .txt should be saved

    Returns:
        Path: Path to the created .txt file
    """

    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("Provided file is not a PDF")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    txt_path = output_dir / f"{pdf_path.stem}.txt"

    reader = PdfReader(pdf_path)

    extracted_text: list[str] = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            extracted_text.append(text)

    txt_path.write_text(
        "\n\n".join(extracted_text),
        encoding="utf-8"
    )

    return txt_path
