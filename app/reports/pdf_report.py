from datetime import datetime
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    HRFlowable,
    Image,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


OUTPUT_FOLDER = Path("reports")


def create_pdf_report(
    prediction: str,
    confidence: float,
    image_path: str,
) -> Path:
    """Create a professional PVision AI inspection report."""

    OUTPUT_FOLDER.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = OUTPUT_FOLDER / f"pvision_inspection_report_{timestamp}.pdf"

    document = SimpleDocTemplate(
        str(report_path),
        rightMargin=1.7 * cm,
        leftMargin=1.7 * cm,
        topMargin=1.5 * cm,
        bottomMargin=1.5 * cm,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "TitleStyle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=26,
        alignment=TA_CENTER,
        textColor=colors.white,
        spaceAfter=8,
    )

    subtitle_style = ParagraphStyle(
        "SubtitleStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#E2E8F0"),
    )

    section_style = ParagraphStyle(
        "SectionStyle",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=14,
        textColor=colors.HexColor("#1E3A8A"),
        spaceBefore=14,
        spaceAfter=8,
    )

    body_style = ParagraphStyle(
        "BodyStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#0F172A"),
    )

    small_style = ParagraphStyle(
        "SmallStyle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#64748B"),
    )

    status = "Healthy" if prediction == "Healthy" else "Defective"

    if status == "Healthy":
        status_bg = colors.HexColor("#DCFCE7")
        status_text = colors.HexColor("#166534")
        risk_level = "Low"
        assessment = (
            "The uploaded photovoltaic panel was classified as <b>HEALTHY</b>. "
            "No visible defect pattern was detected by the AI model."
        )
        recommendation = (
            "Continue routine inspections according to the maintenance schedule. "
            "No immediate corrective action is required based on this image."
        )
    else:
        status_bg = colors.HexColor("#FEE2E2")
        status_text = colors.HexColor("#991B1B")
        risk_level = "Needs Review"
        assessment = (
            "The uploaded photovoltaic panel was classified as <b>DEFECTIVE</b>. "
            "The result should be verified through manual inspection before the panel "
            "is returned to service."
        )
        recommendation = (
            "Perform a detailed visual inspection. Verify electrical performance, "
            "check the affected module area, and replace or isolate the panel if the "
            "damage is confirmed."
        )

    content = []

    header = Table(
        [
            [
                Paragraph("PVision AI", title_style),
            ],
            [
                Paragraph(
                    "AI-powered Photovoltaic Inspection Report",
                    subtitle_style,
                ),
            ],
        ],
        colWidths=[17.5 * cm],
    )

    header.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("BOX", (0, 0), (-1, -1), 0, colors.HexColor("#0F172A")),
                ("TOPPADDING", (0, 0), (-1, -1), 18),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 18),
                ("LEFTPADDING", (0, 0), (-1, -1), 16),
                ("RIGHTPADDING", (0, 0), (-1, -1), 16),
            ]
        )
    )

    content.append(header)
    content.append(Spacer(1, 18))

    content.append(Paragraph("Inspection Summary", section_style))

    summary_table = Table(
        [
            ["Inspection Date", datetime.now().strftime("%d %B %Y, %H:%M")],
            ["Panel Status", status],
            ["Confidence", f"{confidence:.2%}"],
            ["Risk Level", risk_level],
            ["AI Model", "ResNet18 image classifier"],
        ],
        colWidths=[5.2 * cm, 11.8 * cm],
    )

    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F5F9")),
                ("BACKGROUND", (1, 1), (1, 1), status_bg),
                ("TEXTCOLOR", (1, 1), (1, 1), status_text),
                ("TEXTCOLOR", (0, 0), (-1, -1), colors.HexColor("#0F172A")),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 1), (1, 1), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#CBD5E1")),
                ("TOPPADDING", (0, 0), (-1, -1), 9),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 9),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )

    content.append(summary_table)
    content.append(Spacer(1, 14))

    content.append(Paragraph("Confidence Level", section_style))

    confidence_percent = int(confidence * 100)
    filled_width = max(0.4, 16.5 * confidence)

    confidence_bar = Table(
        [["", ""]],
        colWidths=[filled_width * cm, (16.5 - filled_width) * cm],
        rowHeights=[0.45 * cm],
    )

    confidence_bar.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, 0), colors.HexColor("#2563EB")),
                ("BACKGROUND", (1, 0), (1, 0), colors.HexColor("#E2E8F0")),
                ("BOX", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ]
        )
    )

    content.append(confidence_bar)
    content.append(Spacer(1, 4))
    content.append(
        Paragraph(
            f"Model confidence: <b>{confidence_percent}%</b>",
            small_style,
        )
    )

    uploaded_image = Path(image_path)

    if uploaded_image.exists():
        content.append(Paragraph("Uploaded Panel Image", section_style))

        report_image = Image(str(uploaded_image))
        report_image._restrictSize(15.5 * cm, 8.8 * cm)

        image_table = Table(
            [[report_image]],
            colWidths=[17 * cm],
        )

        image_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                    ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 12),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
                ]
            )
        )

        content.append(image_table)

    content.append(Paragraph("AI Assessment", section_style))
    content.append(Paragraph(assessment, body_style))

    content.append(Paragraph("Recommendation", section_style))

    recommendation_box = Table(
        [[Paragraph(recommendation, body_style)]],
        colWidths=[17 * cm],
    )

    recommendation_box.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#F8FAFC")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#CBD5E1")),
                ("LEFTPADDING", (0, 0), (-1, -1), 12),
                ("RIGHTPADDING", (0, 0), (-1, -1), 12),
                ("TOPPADDING", (0, 0), (-1, -1), 12),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 12),
            ]
        )
    )

    content.append(recommendation_box)
    content.append(Spacer(1, 24))

    content.append(
        HRFlowable(
            width="100%",
            thickness=0.8,
            color=colors.HexColor("#CBD5E1"),
        )
    )

    content.append(Spacer(1, 8))

    content.append(
        Paragraph(
            "Generated automatically by PVision AI · Version 0.3",
            small_style,
        )
    )

    document.build(content)

    return report_path