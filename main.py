import pymysql
from db import mysql
from fpdf import FPDF


def download_report():
	conn = None
	cursor = None
	try:
		conn = mysql.connect()
		cursor = conn.cursor(pymysql.cursors.DictCursor)
		
		cursor.execute("SELECT time_interval ,count FROM tbl_count")
		result = cursor.fetchall()
		
		pdf = FPDF()
		pdf.add_page()
		
		page_width = pdf.w - 2 * pdf.l_margin
		
		pdf.set_font('Times','B',14.0) 
		pdf.cell(page_width, 0.0, 'Customer Data', align='C')
		pdf.ln(10)
		pdf.cell(page_width/4, pdf.font_size,'Time Interval')
		pdf.cell(page_width/4, pdf.font_size,'Count')
		pdf.ln(10)
		pdf.set_font('Courier', '', 12)
		
		col_width = page_width/4
		
		pdf.ln(1)
		
		th = pdf.font_size

		for row in result:
			pdf.cell(col_width, th, row['time_interval'], border=1)
			pdf.cell(col_width, th, str(row['count']), border=1)
			pdf.ln(th)
		
		pdf.ln(10)
		
		pdf.set_font('Times','',10.0) 
		pdf.cell(page_width, 0.0, '- end of report -', align='C')
		print('PDF',pdf)
		pdf.output('sample5.pdf', 'F')
	except:
		pass

download_report()