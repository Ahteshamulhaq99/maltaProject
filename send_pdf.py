import pymysql
from db import mysql
from fpdf import FPDF
from datetime import date
from email.mime.application import MIMEApplication
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
import datetime


def download_report(prev_date):
	conn = None
	cursor = None



	# try:
	#### FETCH DATA FROM DATABASE ####
	conn = mysql.connect()
	cursor = conn.cursor(pymysql.cursors.DictCursor)
	cursor.execute(f"SELECT time_interval ,count,average_dwell FROM tbl_count WHERE DATE(`curr_date`) = CURDATE()-1 and cam_id=1 ORDER BY id  DESC")
	result = cursor.fetchall()
	#### CREATE PDF ####
	pdf = FPDF()
	pdf.add_page()
	page_width = pdf.w - 4 * pdf.l_margin
	pdf.set_font('Times','B',26.0)
	title=f'Customer And Staff Report - {prev_date}' 
	pdf.cell(page_width, 0.0, title, align='C')
	pdf.ln(35)
	pdf.set_font('Times','B',14.0)
	pdf.cell(page_width/3, pdf.font_size,'Customer Time Interval',align='C',border=4)
	pdf.cell(page_width/3, pdf.font_size,'Customer Count',align='C',border=4)
	pdf.cell(page_width/3, pdf.font_size,'Customer Average Dwell Time',align='C',border=4)
	pdf.ln(10)
	pdf.set_font('Times', '', 14)
	col_width = page_width/3
	pdf.ln(1)
	th = pdf.font_size
	for row in result:
		if row['time_interval']!='':
			pdf.cell(col_width, th, row['time_interval'], border=1)
			pdf.cell(col_width, th, str(row['count']), border=1)
			pdf.cell(col_width, th, str(row['average_dwell']), border=1)
			pdf.ln(th)




	try:
		cursor = conn.cursor(pymysql.cursors.DictCursor)
		cursor.execute(f"SELECT id, name , count(id) as 'total', sum(dwell_time) as 'total_dwell', avg(dwell_time) as 'average', time_interval FROM tbl_count WHERE DATE(`curr_date`) = CURDATE()-1 and cam_id=2 GROUP BY time_interval ORDER BY id DESC;")
		result = cursor.fetchall()
		pdf.ln(25)
		pdf.set_font('Times','B',14.0)
		pdf.cell(page_width/3, pdf.font_size,'Staff Time Interval',align='C',border=4)
		pdf.cell(page_width/3, pdf.font_size,'Staff Count',align='C',border=4)
		pdf.cell(page_width/3, pdf.font_size,'Staff Average Dwell Time',align='C',border=4)
		pdf.ln(10)
		pdf.set_font('Times', '', 14)
		col_width = page_width/3
		pdf.ln(1)

		for row in result:
			pdf.cell(col_width, th, row['time_interval'], border=1)
			pdf.cell(col_width, th, str(row['total']), border=1)
			pdf.cell(col_width, th, str(row['average']), border=1)
			pdf.ln(th)
		pdf.output('sample.pdf', 'F')
	except:
		pass







def send_email_pdf_figs():
    recipients=['tahanaveed@hotmail.com','umair@viday.ai']
    server = smtplib.SMTP('smtp-mail.outlook.com', 587)
    server.starttls()
    server.login('tahanaveed@hotmail.com', 'N@ust1997')
    msg = MIMEMultipart()
    message = f'Please find attached PDF report for customer Footfall.'
    msg['Subject'] = 'Customer Footfall Report'
    msg['From'] = 'tahanaveed@hotmail.com'
    msg['To'] =', '.join(recipients)
    msg.attach(MIMEText(message, "plain"))
    with open('sample.pdf', "rb") as f:
        attach = MIMEApplication(f.read(),_subtype="pdf")
    attach.add_header('Content-Disposition','attachment',filename=str('sample.pdf'))
    msg.attach(attach)
    server.send_message(msg)
    print("Email Sent!")

# download_report(datetime.datetime.today().date())