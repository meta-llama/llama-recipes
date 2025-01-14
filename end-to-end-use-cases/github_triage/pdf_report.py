from fpdf import FPDF
import os
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())

class ReportPDF(FPDF):
    def __init__(self, repository_name, start_date, end_date):
        FPDF.__init__(self,orientation='P',unit='mm',format='A4')
        self.repo = repository_name
        self.start_end = f"{datetime.strptime(start_date, '%Y-%m-%d').strftime('%b %d, %Y')} to {datetime.strptime(end_date, '%Y-%m-%d').strftime('%b %d, %Y')}"
        
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(100, 10, f'AutoTriage Report: {self.repo}', 0, 0)
        self.cell(90, 10, self.start_end, 0, 0, 'R')
        self.ln(20)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')
        
    def exec_summary(self, text):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, 'Executive Summary', 'B', 0, 'L')
        self.ln(10)
        self.set_font('Arial', '', 10)
        self.multi_cell(0, 5, text)
        self.ln(10)
    
    def add_challenge(self, challenge_data):
        # title
        self.set_font('Arial', '', 14)
        self.cell(0, 10, f"{challenge_data['key_challenge']}", 0, 0, 'L')
        self.ln(8)
        
        # psosible causes
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, "Possible Causes", 0, 0, 'L')
        self.ln(5)
        self.set_font('Arial', '', 10)

        x_list = challenge_data['possible_causes']
        if isinstance(x_list, str):
            x_list = x_list.split(',')

        for x in x_list:
            self.cell(0, 10, "* " + x, 0, 0, 'L')
            self.ln(5)
        self.ln(3)
            
        # remediations
        self.set_font('Arial', 'B', 10)
        self.cell(0, 10, "Remediations", 0, 0, 'L')
        self.ln(5)
        self.set_font('Arial', '', 10)

        x_list = challenge_data['remediations']
        if isinstance(x_list, str):
            x_list = x_list.split(',')

        for x in x_list:
            self.cell(0, 10, "* " + x, 0, 0, 'L')
            self.ln(5)
        self.ln(3)
        
        # affected issues
        self.set_font('Arial', 'B', 10)
        self.cell(30, 10, f"Affected issues: ", 0, 0, 'L')
        
        x_list = challenge_data['affected_issues']
        if isinstance(x_list, str):
            x_list = x_list.split(',')
            
        for iss in x_list:
            self.set_text_color(0,0,255)
            self.cell(12, 10, str(iss), 0, 0, 'L', link=f"https://github.com/{self.repo}/issues/{iss}")
            
        self.set_text_color(0,0,0)
        self.ln(15)

    def challenges_section(self, key_challenges_data):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, 'Key Challenges', 'B', 0, 'L')
        self.ln(10)
        for cd in key_challenges_data:
            self.add_challenge(cd)
        self.ln(20)
    
    def open_ques_section(self, open_questions):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, 'Open Questions', 'B', 0, 'L')
        self.ln(10)
        self.set_font('Arial', '', 10)

        if isinstance(open_questions, str):
            open_questions = open_questions.split(',')
                    
        for qq in open_questions:
            self.multi_cell(0, 5, "* " + qq, 0, 0, 'L')
            self.ln(5)
        self.ln(5)
    
    def add_graphs_section(self, title, plot_paths):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 8, f'[Viz] {title}', 'B', 0, 'L')
        self.ln(10)
        for path in plot_paths:
            if os.path.exists(path):
                self.add_plot(path)
            else:
                self.set_font('Arial', 'BI', 10)
                self.cell(0, 8, '< Plot not found, make sure you have push-acces to this repo >', 0, 0)
        self.ln(10)
            
    def add_plot(self, img):
        self.image(img, x=30, w=150)
        self.ln(5)
        
    
    
def create_report_pdf(repo_name, start_date, end_date, key_challenges_data, executive_summary, open_questions, out_folder):#, image1, image2):
    out_path = f'{out_folder}/report.pdf'
    logger.info(f"Creating PDF report at {out_path}")
    
    pdf = ReportPDF(repo_name, start_date, end_date)
    pdf.add_page()
    pdf.exec_summary(executive_summary)
    pdf.open_ques_section(open_questions)
    pdf.challenges_section(key_challenges_data)
    pdf.add_page()
    pdf.add_graphs_section("Repo Maintenance", [f'{out_folder}/plots/engagement_sankey.png'])
    pdf.add_page()
    pdf.add_graphs_section("Traffic in the last 2 weeks", [f'{out_folder}/plots/{x}.png' for x in ['views_clones','resources', 'referrers']])
    pdf.add_page()
    pdf.add_graphs_section("New issues in the last 2 weeks", [f'{out_folder}/plots/{x}.png' for x in ['themes', 'severity', 'sentiment', 'expertise']])
    pdf.output(out_path, 'F')

