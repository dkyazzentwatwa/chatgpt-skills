#!/usr/bin/env python3
"""
Resume Builder - Generate professional PDF resumes

Features:
- Multiple templates (modern, classic, minimal, executive)
- ATS-friendly formatting
- Structured data input (JSON or API)
- Customizable sections
"""

import argparse
import json
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    ListFlowable, ListItem, HRFlowable
)


@dataclass
class ContactInfo:
    name: str
    email: str = ""
    phone: str = ""
    location: str = ""
    linkedin: str = ""
    github: str = ""
    website: str = ""


@dataclass
class Experience:
    title: str
    company: str
    dates: str
    bullets: List[str] = field(default_factory=list)
    location: str = ""


@dataclass
class Education:
    degree: str
    school: str
    year: str
    gpa: str = ""
    honors: str = ""
    coursework: List[str] = field(default_factory=list)


@dataclass
class Project:
    name: str
    description: str
    technologies: List[str] = field(default_factory=list)
    url: str = ""


@dataclass
class Certification:
    name: str
    issuer: str
    year: str


class ResumeBuilder:
    """
    Generate professional PDF resumes.

    Example:
        resume = ResumeBuilder()
        resume.set_contact("John Doe", "john@email.com")
        resume.add_experience("Engineer", "Company", "2020-Present", [...])
        resume.generate().save("resume.pdf")
    """

    def __init__(self, template: str = "modern"):
        """Initialize resume builder."""
        self._template = template
        self._contact: Optional[ContactInfo] = None
        self._summary = ""
        self._objective = ""
        self._experience: List[Experience] = []
        self._education: List[Education] = []
        self._skills: Union[List[str], Dict[str, List[str]]] = []
        self._projects: List[Project] = []
        self._certifications: List[Certification] = []
        self._languages: List[str] = []
        self._volunteer: List[Dict] = []
        self._publications: List[Dict] = []
        self._custom_sections: Dict[str, List[str]] = {}

        # Styling
        self._primary_color = "#2563eb"
        self._text_color = "#333333"
        self._margins = {"top": 0.5, "bottom": 0.5, "left": 0.6, "right": 0.6}

        self._pdf_buffer: Optional[BytesIO] = None

    @classmethod
    def from_json(cls, filepath: str) -> 'ResumeBuilder':
        """Load resume from JSON file."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        with open(filepath, 'r') as f:
            data = json.load(f)

        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ResumeBuilder':
        """Load resume from dictionary."""
        resume = cls()

        # Contact
        if 'contact' in data:
            c = data['contact']
            resume.set_contact(
                name=c.get('name', ''),
                email=c.get('email', ''),
                phone=c.get('phone', ''),
                location=c.get('location', ''),
                linkedin=c.get('linkedin', ''),
                github=c.get('github', ''),
                website=c.get('website', '')
            )

        # Summary/Objective
        if 'summary' in data:
            resume.set_summary(data['summary'])
        if 'objective' in data:
            resume.set_objective(data['objective'])

        # Experience
        for exp in data.get('experience', []):
            resume.add_experience(
                title=exp.get('title', ''),
                company=exp.get('company', ''),
                dates=exp.get('dates', ''),
                bullets=exp.get('bullets', []),
                location=exp.get('location', '')
            )

        # Education
        for edu in data.get('education', []):
            resume.add_education(
                degree=edu.get('degree', ''),
                school=edu.get('school', ''),
                year=edu.get('year', ''),
                gpa=edu.get('gpa', ''),
                honors=edu.get('honors', ''),
                coursework=edu.get('coursework', [])
            )

        # Skills
        if 'skills' in data:
            resume.add_skills(data['skills'])

        # Projects
        for proj in data.get('projects', []):
            resume.add_project(
                name=proj.get('name', ''),
                description=proj.get('description', ''),
                technologies=proj.get('technologies', []),
                url=proj.get('url', '')
            )

        # Certifications
        for cert in data.get('certifications', []):
            resume.add_certification(
                name=cert.get('name', ''),
                issuer=cert.get('issuer', ''),
                year=cert.get('year', '')
            )

        # Languages
        if 'languages' in data:
            resume.add_languages(data['languages'])

        return resume

    def set_template(self, template: str) -> 'ResumeBuilder':
        """Set resume template."""
        valid = ['modern', 'classic', 'minimal', 'executive']
        if template not in valid:
            raise ValueError(f"Invalid template: {template}. Use {valid}")
        self._template = template
        return self

    def set_contact(
        self,
        name: str,
        email: str = "",
        phone: str = "",
        location: str = "",
        linkedin: str = "",
        github: str = "",
        website: str = ""
    ) -> 'ResumeBuilder':
        """Set contact information."""
        self._contact = ContactInfo(
            name=name,
            email=email,
            phone=phone,
            location=location,
            linkedin=linkedin,
            github=github,
            website=website
        )
        return self

    def set_summary(self, summary: str) -> 'ResumeBuilder':
        """Set professional summary."""
        self._summary = summary
        return self

    def set_objective(self, objective: str) -> 'ResumeBuilder':
        """Set career objective."""
        self._objective = objective
        return self

    def add_experience(
        self,
        title: str,
        company: str,
        dates: str,
        bullets: List[str],
        location: str = ""
    ) -> 'ResumeBuilder':
        """Add work experience entry."""
        exp = Experience(
            title=title,
            company=company,
            dates=dates,
            bullets=bullets,
            location=location
        )
        self._experience.append(exp)
        return self

    def add_education(
        self,
        degree: str,
        school: str,
        year: str,
        gpa: str = "",
        honors: str = "",
        coursework: List[str] = None
    ) -> 'ResumeBuilder':
        """Add education entry."""
        edu = Education(
            degree=degree,
            school=school,
            year=year,
            gpa=gpa,
            honors=honors,
            coursework=coursework or []
        )
        self._education.append(edu)
        return self

    def add_skills(
        self,
        skills: Union[List[str], Dict[str, List[str]]]
    ) -> 'ResumeBuilder':
        """Add skills (list or categorized dict)."""
        self._skills = skills
        return self

    def add_project(
        self,
        name: str,
        description: str,
        technologies: List[str] = None,
        url: str = ""
    ) -> 'ResumeBuilder':
        """Add project entry."""
        proj = Project(
            name=name,
            description=description,
            technologies=technologies or [],
            url=url
        )
        self._projects.append(proj)
        return self

    def add_certification(
        self,
        name: str,
        issuer: str,
        year: str
    ) -> 'ResumeBuilder':
        """Add certification."""
        cert = Certification(name=name, issuer=issuer, year=year)
        self._certifications.append(cert)
        return self

    def add_languages(self, languages: List[str]) -> 'ResumeBuilder':
        """Add language proficiencies."""
        self._languages = languages
        return self

    def add_volunteer(
        self,
        role: str,
        organization: str,
        dates: str,
        description: str = ""
    ) -> 'ResumeBuilder':
        """Add volunteer experience."""
        self._volunteer.append({
            'role': role,
            'organization': organization,
            'dates': dates,
            'description': description
        })
        return self

    def add_publication(
        self,
        title: str,
        venue: str,
        year: str,
        url: str = ""
    ) -> 'ResumeBuilder':
        """Add publication."""
        self._publications.append({
            'title': title,
            'venue': venue,
            'year': year,
            'url': url
        })
        return self

    def add_custom_section(
        self,
        title: str,
        items: List[str]
    ) -> 'ResumeBuilder':
        """Add custom section."""
        self._custom_sections[title] = items
        return self

    def set_colors(
        self,
        primary: str = None,
        text: str = None
    ) -> 'ResumeBuilder':
        """Set color scheme."""
        if primary:
            self._primary_color = primary
        if text:
            self._text_color = text
        return self

    def set_margins(
        self,
        top: float = None,
        bottom: float = None,
        left: float = None,
        right: float = None
    ) -> 'ResumeBuilder':
        """Set page margins in inches."""
        if top:
            self._margins['top'] = top
        if bottom:
            self._margins['bottom'] = bottom
        if left:
            self._margins['left'] = left
        if right:
            self._margins['right'] = right
        return self

    def _hex_to_color(self, hex_color: str):
        """Convert hex to reportlab color."""
        hex_color = hex_color.lstrip('#')
        r = int(hex_color[0:2], 16) / 255
        g = int(hex_color[2:4], 16) / 255
        b = int(hex_color[4:6], 16) / 255
        return colors.Color(r, g, b)

    def generate(self) -> 'ResumeBuilder':
        """Generate the resume PDF."""
        self._pdf_buffer = BytesIO()

        doc = SimpleDocTemplate(
            self._pdf_buffer,
            pagesize=letter,
            rightMargin=self._margins['right'] * inch,
            leftMargin=self._margins['left'] * inch,
            topMargin=self._margins['top'] * inch,
            bottomMargin=self._margins['bottom'] * inch
        )

        elements = []
        styles = getSampleStyleSheet()

        primary = self._hex_to_color(self._primary_color)
        text_color = self._hex_to_color(self._text_color)

        # Template-specific fonts
        if self._template in ['classic', 'executive']:
            header_font = "Times-Bold"
            body_font = "Times-Roman"
        else:
            header_font = "Helvetica-Bold"
            body_font = "Helvetica"

        # Custom styles
        name_style = ParagraphStyle(
            'Name',
            fontName=header_font,
            fontSize=20 if self._template != 'minimal' else 16,
            textColor=primary if self._template != 'minimal' else text_color,
            alignment=TA_CENTER,
            spaceAfter=4
        )

        contact_style = ParagraphStyle(
            'Contact',
            fontName=body_font,
            fontSize=10,
            textColor=text_color,
            alignment=TA_CENTER,
            spaceAfter=12
        )

        section_style = ParagraphStyle(
            'Section',
            fontName=header_font,
            fontSize=12,
            textColor=primary if self._template != 'minimal' else text_color,
            spaceBefore=12,
            spaceAfter=6
        )

        job_title_style = ParagraphStyle(
            'JobTitle',
            fontName=header_font,
            fontSize=11,
            textColor=text_color
        )

        company_style = ParagraphStyle(
            'Company',
            fontName=body_font,
            fontSize=10,
            textColor=text_color
        )

        body_style = ParagraphStyle(
            'Body',
            fontName=body_font,
            fontSize=10,
            textColor=text_color,
            alignment=TA_JUSTIFY
        )

        bullet_style = ParagraphStyle(
            'Bullet',
            fontName=body_font,
            fontSize=10,
            textColor=text_color,
            leftIndent=15,
            bulletIndent=5
        )

        # Header / Contact
        if self._contact:
            elements.append(Paragraph(self._contact.name, name_style))

            contact_parts = []
            if self._contact.email:
                contact_parts.append(self._contact.email)
            if self._contact.phone:
                contact_parts.append(self._contact.phone)
            if self._contact.location:
                contact_parts.append(self._contact.location)

            if contact_parts:
                elements.append(Paragraph(" | ".join(contact_parts), contact_style))

            # Links
            link_parts = []
            if self._contact.linkedin:
                link_parts.append(self._contact.linkedin)
            if self._contact.github:
                link_parts.append(self._contact.github)
            if self._contact.website:
                link_parts.append(self._contact.website)

            if link_parts:
                elements.append(Paragraph(" | ".join(link_parts), contact_style))

        # Divider line
        if self._template != 'minimal':
            elements.append(HRFlowable(
                width="100%",
                thickness=1,
                color=primary,
                spaceBefore=6,
                spaceAfter=6
            ))

        # Summary/Objective
        if self._summary or self._objective:
            text = self._summary or self._objective
            label = "PROFESSIONAL SUMMARY" if self._summary else "OBJECTIVE"
            elements.append(Paragraph(label, section_style))
            elements.append(Paragraph(text, body_style))

        # Experience
        if self._experience:
            elements.append(Paragraph("EXPERIENCE", section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

            for exp in self._experience:
                # Title and dates on same line
                title_date = Table(
                    [[Paragraph(f"<b>{exp.title}</b>", job_title_style),
                      Paragraph(exp.dates, company_style)]],
                    colWidths=[4.5 * inch, 2.5 * inch]
                )
                title_date.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ]))
                elements.append(title_date)

                # Company and location
                company_loc = exp.company
                if exp.location:
                    company_loc += f" - {exp.location}"
                elements.append(Paragraph(company_loc, company_style))

                # Bullets
                for bullet in exp.bullets:
                    elements.append(Paragraph(f"• {bullet}", bullet_style))

                elements.append(Spacer(1, 6))

        # Education
        if self._education:
            elements.append(Paragraph("EDUCATION", section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

            for edu in self._education:
                # Degree and year
                deg_year = Table(
                    [[Paragraph(f"<b>{edu.degree}</b>", job_title_style),
                      Paragraph(edu.year, company_style)]],
                    colWidths=[5 * inch, 2 * inch]
                )
                deg_year.setStyle(TableStyle([
                    ('ALIGN', (0, 0), (0, 0), 'LEFT'),
                    ('ALIGN', (1, 0), (1, 0), 'RIGHT'),
                ]))
                elements.append(deg_year)
                elements.append(Paragraph(edu.school, company_style))

                if edu.gpa:
                    elements.append(Paragraph(f"GPA: {edu.gpa}", body_style))
                if edu.honors:
                    elements.append(Paragraph(edu.honors, body_style))
                if edu.coursework:
                    elements.append(Paragraph(
                        f"Relevant Coursework: {', '.join(edu.coursework)}",
                        body_style
                    ))

                elements.append(Spacer(1, 6))

        # Skills
        if self._skills:
            elements.append(Paragraph("SKILLS", section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

            if isinstance(self._skills, dict):
                for category, skills in self._skills.items():
                    elements.append(Paragraph(
                        f"<b>{category}:</b> {', '.join(skills)}",
                        body_style
                    ))
            else:
                elements.append(Paragraph(', '.join(self._skills), body_style))

        # Projects
        if self._projects:
            elements.append(Paragraph("PROJECTS", section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

            for proj in self._projects:
                proj_text = f"<b>{proj.name}</b>"
                if proj.url:
                    proj_text += f" ({proj.url})"
                elements.append(Paragraph(proj_text, job_title_style))
                elements.append(Paragraph(proj.description, body_style))
                if proj.technologies:
                    elements.append(Paragraph(
                        f"Technologies: {', '.join(proj.technologies)}",
                        body_style
                    ))
                elements.append(Spacer(1, 4))

        # Certifications
        if self._certifications:
            elements.append(Paragraph("CERTIFICATIONS", section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))

            for cert in self._certifications:
                elements.append(Paragraph(
                    f"<b>{cert.name}</b> - {cert.issuer} ({cert.year})",
                    body_style
                ))

        # Languages
        if self._languages:
            elements.append(Paragraph("LANGUAGES", section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            elements.append(Paragraph(', '.join(self._languages), body_style))

        # Custom sections
        for title, items in self._custom_sections.items():
            elements.append(Paragraph(title.upper(), section_style))
            if self._template != 'minimal':
                elements.append(HRFlowable(width="100%", thickness=0.5, color=colors.lightgrey))
            for item in items:
                elements.append(Paragraph(f"• {item}", bullet_style))

        # Build PDF
        doc.build(elements)

        return self

    def save(self, path: str) -> str:
        """Save resume to PDF file."""
        if self._pdf_buffer is None:
            self.generate()

        output_path = Path(path)
        with open(output_path, 'wb') as f:
            f.write(self._pdf_buffer.getvalue())

        return str(output_path)

    def to_bytes(self) -> bytes:
        """Get PDF as bytes."""
        if self._pdf_buffer is None:
            self.generate()
        return self._pdf_buffer.getvalue()


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Generate professional PDF resumes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python resume_builder.py --input resume.json --output resume.pdf
  python resume_builder.py --input data.json --template modern --output resume.pdf
        """
    )

    parser.add_argument('--input', '-i', required=True, help='Input JSON file')
    parser.add_argument('--output', '-o', default='resume.pdf', help='Output PDF path')
    parser.add_argument('--template', '-t', default='modern',
                        choices=['modern', 'classic', 'minimal', 'executive'],
                        help='Template style')

    args = parser.parse_args()

    try:
        resume = ResumeBuilder.from_json(args.input)
        resume.set_template(args.template)
        resume.generate().save(args.output)
        print(f"Resume saved to: {args.output}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
