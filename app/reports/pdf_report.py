from datetime import datetime
from pathlib import Path

from reportlab.lib.styles import getSampleStyleSheet


OUTPUT_FOLDER = Path("reports")
OUTPUT_FOLDER.mkdir(exist_ok=True)


def create_pdf_report(
    prediction: str,
    confidence: float,
) -> Path:
    """Create an inspection report as a PDF."""

    report_path = OUTPUT_FOLDER / "inspection_report.pdf"

    document = SimpleDocTemplate(str(report_path))
    styles = getSampleStyleSheet()

    content = []

    content.append(Paragraph("<b>PVision AI</b>", styles["Title"]))
    content.append(
        Paragraph(
            "Solar Panel Inspection Report",
            styles["Heading2"],
        )
    )

    content.append(
        Paragraph(
            f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            styles["Normal"],
        )
    )

    content.append(
        Paragraph(
            f"<b>Panel Status:</b> {prediction}",
            styles["Normal"],
        )
    )

    content.append(
        Paragraph(
            f"<b>Confidence:</b> {confidence:.2%}",
            styles["Normal"],
        )
    )

    if prediction == "Healthy":
        recommendation = (
            "No visible defects were detected. "
            "Continue with routine inspections."
        )
    else:
        recommendation = (
            "A defect was detected. "
            "A manual inspection is recommended."
        )

    content.append(
        Paragraph(
            f"<b>Recommendation:</b> {recommendation}",
            styles["Normal"],
        )
    )

    document.build(content)

    return report_path
