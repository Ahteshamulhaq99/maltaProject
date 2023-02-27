import mysql.connector


class Database:
    
    def __init__(self) -> None:
        pass

    def insert_staff(self, cam_id,id,img_val,entry_time,exit_time,person_time):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        person_time=str(person_time)+''+'seconds'
        val=(cam_id,id,img_val,entry_time,exit_time,person_time)
        mycursor = mydb.cursor()
        sql = "INSERT INTO tbl_staff (cam_id,person_id,image_path, enter_time, exit_time, dwell_time) VALUES (%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()

    def insert_customer(self, cam_id,id,img_val,entry_time,exit_time,person_time):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        person_time=str(person_time)+''+'seconds'
        val=(cam_id,id,img_val,entry_time,exit_time,person_time)
        mycursor = mydb.cursor()
        sql = "INSERT INTO tbl_customer (cam_id,person_id,image_path, enter_time, exit_time, dwell_time) VALUES (%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()

    def insert_count(self,cam_id,curr_day,time_interval,person_count,average_dwell,curr_time):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        val=(cam_id,curr_day,time_interval,person_count,average_dwell,curr_time)
        mycursor = mydb.cursor()
        sql = "INSERT INTO tbl_count (cam_id,curr_date,time_interval,count,average_dwell,curr_time) VALUES (%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql, val)
        mydb.commit()
        print("CUSTOMER DATA PUSHEDDDDDDDDDDDDDDDDDDDDDDDDDDDDD")
        mycursor.close()
        mydb.close()

    def insert_count_staff(self,cam_id,dwell_time,curr_day,time_interval,person_id,curr_time):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        val=(cam_id,curr_day,time_interval,dwell_time,person_id,curr_time)
        mycursor = mydb.cursor()
        sql = "INSERT INTO tbl_count (cam_id,curr_date,time_interval,dwell_time,person_id,curr_time) VALUES (%s,%s,%s,%s,%s,%s)"
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()

    def update_dwell(self,cam_id,dwell_time,person_id):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        mycursor = mydb.cursor()
        sql = f"UPDATE tbl_count SET dwell_time={dwell_time} WHERE cam_id=2 and person_id={person_id};"
        mycursor.execute(sql)
        mydb.commit()
        print("DWELL TIME UPDATEDDDDDDDDDDDDDDDDDDDDDD")
        mycursor.close()
        mydb.close()
    
    def update_absent(self,cam_id,curr_date,curr_time,val):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        val=(int(cam_id),curr_date,curr_time,val)
        mycursor = mydb.cursor()
        sql = "INSERT INTO tbl_count (cam_id,curr_date,curr_time,absent) VALUES (%s,%s,%s,%s)"
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()

    def update_blackout(self,cam_id,curr_date,curr_time,blackout_start,blackout_end):
        mydb = mysql.connector.connect(
            host="localhost",
            user="root",
            password="",
            database="malta_db"
            )
        val=(int(cam_id),curr_date,curr_time,blackout_start,blackout_end)
        mycursor = mydb.cursor()
        sql = "INSERT INTO tbl_count (cam_id,curr_date,curr_time,blackout_start,blackout_end) VALUES (%s,%s,%s,%s,%s)"
        mycursor.execute(sql, val)
        mydb.commit()
        mycursor.close()
        mydb.close()