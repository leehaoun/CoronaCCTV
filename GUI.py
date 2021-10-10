import datetime
import sys
import pymysql
from PySide6.QtCore import QDateTime, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QApplication, QMessageBox, QTableWidgetItem
from qt_material import apply_stylesheet

import hanium_detect
from ui import Ui_MainWindow


#  pyside6, qt_material, pymysql 설치 필요 (requirements에 추가했음)
#  폴더 내의 material.css.template을 파이썬이 설치된 폴더의 Lib\site-packages\qt_material 안에 붙여넣어야 한다.
# TODO 기능 추가

class MainWindow(QMainWindow):
    # DB 연결
    corona_db = pymysql.connect(
        user='root',
        passwd='root',
        host='localhost',
        db='corona',
        charset='utf8mb4'
    )
    cursor = corona_db.cursor(pymysql.cursors.DictCursor)

    # 코로나 CCTV 프로그램 실행 시 넘겨줄 파라미터값 저장 변수
    processor = "GPU"
    label = "ON"
    rel = "50"
    display = "1280x720"
    alarm = "모두"
    corona_act = "모두"

    # Checkbox 선택 시 무한 호출 방지용 변수 + 시간 변수 + 로그용 인덱스 변수
    lock_alarm = False
    lock_corona = False
    datetime = QDateTime.currentDateTime()
    index = 0

    def __init__(self):

        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # DB 테이블 시간 컬럼 크기 수정
        self.ui.table_db.setColumnWidth(0, self.width() * 2 / 10)

        # 아이콘 설정
        self.setWindowIcon(QIcon("./cctv.png"))

        # 프로그램 첫 실행 시 DB 생성
        db_check = "SHOW TABLES LIKE 'log';"
        self.cursor.execute(db_check)
        db_check_result = self.cursor.fetchall()
        if len(db_check_result) == 0:
            self.cursor.execute("""CREATE TABLE log(
        id INT(255) NOT NULL AUTO_INCREMENT PRIMARY KEY, 
        time DATETIME, 
        check_act VARCHAR(255),
        ab_path VARCHAR(255),
        file_name VARCHAR(255)
        );""")

        # GUI 실행 시 DB 정보 불러오기
        db_init = """SELECT * FROM log"""
        self.cursor.execute(db_init)
        db_init_data = self.cursor.fetchall()
        db_sani_num = 0
        db_temp_num = 0
        db_mask_num = 0
        db_qr_num = 0
        for data in db_init_data:
            index = 0
            self.ui.table_db.insertRow(index)
            time_data = data['time'].strftime('%Y-%m-%d %H:%M:%S')
            self.ui.table_db.setItem(index, 0, QTableWidgetItem(time_data))
            self.ui.table_db.setItem(index, 1, QTableWidgetItem(data['check_act']))
            index += 1
            if data['check_act'] == '손소독':
                db_sani_num += 1
            elif data['check_act'] == '온도계':
                db_temp_num += 1
            elif data['check_act'] == 'QR':
                db_qr_num += 1
            else:
                db_mask_num += 1

        db_total_num = db_sani_num + db_temp_num + db_qr_num + db_mask_num
        self.ui.lcd_db_sani.display(db_sani_num)
        self.ui.lcd_db_temp.display(db_temp_num)
        self.ui.lcd_db_qr.display(db_qr_num)
        self.ui.lcd_db_mask.display(db_mask_num)
        self.ui.lcd_db_total.display(db_total_num)

        # 시간 출력, self.tick 호출, DB 시간 실행 시간으로 맞춤
        now = datetime.datetime.now()
        self.ui.time_db.setDateTime(now)

        self.timer = QTimer()
        self.timer.setInterval(990)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        # 설명 버튼 누를 시 self.explain 호출
        self.ui.btn_explain_processor.clicked.connect(self.explain)
        self.ui.btn_explain_display.clicked.connect(self.explain)
        self.ui.btn_explain_alarm.clicked.connect(self.explain)
        self.ui.btn_explain_check.clicked.connect(self.explain)
        self.ui.btn_explain_label.clicked.connect(self.explain)
        self.ui.btn_explain_reliability.clicked.connect(self.explain)

        # 신뢰도 설정 시 self.sync_rel 호출
        self.ui.spin_rel.valueChanged.connect(self.sync_rel)
        self.ui.hs_rel.valueChanged.connect(self.sync_rel)

        # 경보 버튼 선택 시 self.alarm_check 호출
        self.ui.alarm_ck_all.stateChanged.connect(self.alarm_check)
        self.ui.alarm_ck_msg.stateChanged.connect(self.alarm_check)
        self.ui.alarm_ck_siren.stateChanged.connect(self.alarm_check)
        self.ui.alarm_ck_light.stateChanged.connect(self.alarm_check)

        # 검사할 코로나 소독 행위 버튼 선택 시 self.corona_check 호출
        self.ui.corona_ck_all.stateChanged.connect(self.corona_check)
        self.ui.corona_ck_sani.stateChanged.connect(self.corona_check)
        self.ui.corona_ck_temp.stateChanged.connect(self.corona_check)
        self.ui.corona_ck_qr.stateChanged.connect(self.corona_check)

        # 프로세서 선택 시 self.processor_check 호출
        self.ui.radio_cpu.toggled.connect(self.processor_check)
        self.ui.radio_gpu.toggled.connect(self.processor_check)

        # 라벨 유무 선택 시 self.label_check 호출
        self.ui.radio_on.toggled.connect(self.label_check)
        self.ui.radio_off.toggled.connect(self.label_check)

        # 해상도 선택 시 self.display_check 호출
        self.ui.cb_display.currentIndexChanged.connect(self.display_check)

        # UI 실행 시 설정환경창 업데이트
        self.text_load()

        # 로그 초기화 버튼 누를 시 log_clear 호출 + DB 초기화 버튼 누를 시 db_clear 호출
        self.ui.btn_log_clear.clicked.connect(self.log_clear)
        self.ui.btn_db_clear.clicked.connect(self.db_clear)

        # 원래대로 버튼이나 DB 업데이트 버튼 누를 시 db_update 호출
        self.ui.btn_db_update.clicked.connect(self.db_update)
        self.ui.btn_db_ori.clicked.connect(self.db_update)

        # DB 검색 버튼 누를 시 db_search 호출
        self.ui.btn_search_db.clicked.connect(self.db_search)

        # 시작 버튼 누를 시 self.start 호출
        self.ui.btn_start.clicked.connect(self.start)

    # 시간 출력 함수
    def tick(self):
        now = datetime.datetime.now()
        df = '%Y년 %#m월 %#d일 %H:%M:%S'
        self.ui.lb_now.setText(f"현재시각 : {now.strftime(df)}")

    # 설명 버튼 누를 시 메세지 박스 띄우는 함수
    def explain(self):
        msg = QMessageBox()
        msg.setWindowTitle("설명")
        msg.setIcon(QMessageBox.Information)
        msg.setWindowIcon(QIcon("./cctv.png"))

        if self.sender() == self.ui.btn_explain_processor:
            msg.setText(
                "어떤 프로세서를 활용하여 프로그램을 실행할지 선택하세요\n" +
                "단, CPU를 선택할 시 이미지 검사만 가능합니다"
            )
        elif self.sender() == self.ui.btn_explain_label:
            msg.setText(
                "어떤 사물이 포착되었는지 텍스트로 표시할지 정하세요\n"
            )
        elif self.sender() == self.ui.btn_explain_alarm:
            msg.setText(
                "어떤 경보를 사용할지 선택하세요"
            )
        elif self.sender() == self.ui.btn_explain_display:
            msg.setText(
                "실행 화면의 크기를 정하세요"
            )
        elif self.sender() == self.ui.btn_explain_reliability:
            msg.setText(
                "신회도가 높을 수록 확실한 사물만 탐지합니다\n" +
                "단, 신뢰도가 너무 높을 시 탐지를 못할 수 있습니다 (50% 권장)"
            )
        elif self.sender() == self.ui.btn_explain_check:
            msg.setText(
                "어떤 소독 행위에 대해 검사를 실시할지 선택하세요"
            )

        msg.exec()

    # 신뢰도 두 버튼 싱크 맞추기 + rel 변수 업데이트 + 설정환경창 업데이트
    def sync_rel(self):
        if self.sender() == self.ui.spin_rel:
            self.ui.hs_rel.setValue(self.ui.spin_rel.value())
        elif self.sender() == self.ui.hs_rel:
            self.ui.spin_rel.setValue(self.ui.hs_rel.value())

        self.rel = self.ui.spin_rel.value()
        self.text_load()

    # 경보 버튼 선택 시 동작
    # TODO 딕셔너리로 수정, 설정환경창 업데이트, 파라미터값 업데이트
    def alarm_check(self):
        if self.sender() == self.ui.alarm_ck_all and self.lock_alarm is False:
            self.ui.alarm_ck_light.setChecked(self.ui.alarm_ck_all.isChecked())
            self.ui.alarm_ck_msg.setChecked(self.ui.alarm_ck_all.isChecked())
            self.ui.alarm_ck_siren.setChecked(self.ui.alarm_ck_all.isChecked())
        elif self.sender().isChecked() is False:
            self.lock_alarm = True
            self.ui.alarm_ck_all.setChecked(False)
        else:
            self.lock_alarm = True
            if self.ui.alarm_ck_light.isChecked() and self.ui.alarm_ck_siren.isChecked() and \
                    self.ui.alarm_ck_msg.isChecked():
                self.ui.alarm_ck_all.setChecked(True)

        alarm_widgets = [
            "alarm_ck_all",
            "alarm_ck_siren",
            "alarm_ck_light",
            "alarm_ck_msg",
        ]
        self.alarm = ""
        for w in alarm_widgets:
            widget = getattr(self.ui, w)
            if widget.isChecked() and widget.text() == '모두':
                self.alarm = "모두"
                break
            elif widget.isChecked():
                self.alarm += widget.text()
                self.alarm += " "
        self.text_load()
        self.lock_alarm = False

    # 검사할 코로나 소독 행위 선택 시 동작 + 파라미터값, 설정환경창 업데이트
    def corona_check(self):
        if self.sender() == self.ui.corona_ck_all and self.lock_corona is False:
            self.ui.corona_ck_qr.setChecked(self.ui.corona_ck_all.isChecked())
            self.ui.corona_ck_temp.setChecked(self.ui.corona_ck_all.isChecked())
            self.ui.corona_ck_sani.setChecked(self.ui.corona_ck_all.isChecked())
        elif self.sender().isChecked() is False:
            self.lock_corona = True
            self.ui.corona_ck_all.setChecked(False)
        else:
            self.lock_corona = True
            if self.ui.corona_ck_qr.isChecked() and self.ui.corona_ck_temp.isChecked() \
                    and self.ui.corona_ck_sani.isChecked():
                self.ui.corona_ck_all.setChecked(True)

        corona_act_widgets = [
            "corona_ck_all",
            "corona_ck_sani",
            "corona_ck_temp",
            "corona_ck_qr",
        ]
        self.corona_act = ""
        for w in corona_act_widgets:
            widget = getattr(self.ui, w)
            if widget.isChecked() and widget.text() == '모두':
                self.corona_act = "모두"
                break
            elif widget.isChecked():
                self.corona_act += widget.text()
                self.corona_act += " "
        self.text_load()
        self.lock_corona = False

    # 프로세서 선택 시 동작 : processor 변수 업데이트 + 설정환경창 업데이트
    def processor_check(self):
        if self.sender().isChecked() is True:
            self.processor = self.sender().text()
            self.text_load()

    # 라벨 유무 선택 시 동작 : label 변수 업데이트 + 설정환경창 업데이트
    def label_check(self):
        if self.sender().isChecked() is True:
            self.label = self.sender().text()
            self.text_load()

    # 해상도 선택 시 동작 : display 변수 업데이트 + 설정환경창 업데이트
    def display_check(self):
        self.display = self.sender().currentText()
        self.text_load()

    # 설정환경창 설정
    # TODO 경보, 코로나 소독 행위 추가
    def text_load(self):
        self.ui.text_status.setText(
            f"프로세서: {self.processor}\n" +
            f"라벨 표시 유무: {self.label}\n" +
            f"경보: {self.alarm}\n" +
            f"소독: {self.corona_act}\n"
            f"화면 해상도: {self.display}\n" +
            f"신뢰도: {self.rel}%"
        )

    # 시작 버튼 누를 시 동작 : 파라미터값 업데이트 + lcd 업데이트 + 코로나 CCTV 실행
    def start(self):
        # 파라미터값 업데이트 + DB 연결
        if self.processor == "GPU":
            device = '0'
        else:
            device = 'cpu'

        if self.label == 'ON':
            hide_labels = False
        else:
            hide_labels = True

        if self.corona_act == "":
            msg = QMessageBox()
            msg.warning(self, '경고!', "검사할 코로나 소독 행위를 선택해주세요!")
            return 
        elif self.corona_act == "모두":
            mod_set = 0
        elif self.corona_act == "온도계 QR 체크 ":
            mod_set = 1
        elif self.corona_act == '손 소독 QR 체크 ':
            mod_set = 2
        elif self.corona_act == '손 소독 온도계 ':
            mod_set = 3
        elif self.corona_act == '손 소독 ':
            mod_set = 4
        elif self.corona_act == "온도계 ":
            mod_set = 5
        else:
            mod_set = 6

        if self.alarm == "":
            msg = QMessageBox()
            reply = msg.warning(self, '경고!', "어떤 경보도 선택되지 않았습니다!\n 그래도 진행하시겠습니까?",
                                QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.No:
                return
            else:
                alarm = 7
        elif self.alarm == "모두":
            alarm = 0
        elif self.alarm == "경광등 메세지 ":
            alarm = 1
        elif self.alarm == "사이렌 메세지 ":
            alarm = 2
        elif self.alarm == "사이렌 경광등 ":
            alarm = 3
        elif self.alarm == "사이렌 ":
            alarm = 4
        elif self.alarm == "경광등 ":
            alarm = 5
        else:
            alarm = 6

        win_size = self.display.split('x')

        # lcd + log 업데이트
        start_timer = QTimer()
        start_timer.setInterval(990)
        start_timer.timeout.connect(self.lcd_update)
        start_timer.timeout.connect(self.log_update)
        start_timer.start()

        # 코로나 CCTV 실행
        hanium_detect.detect(source='C:/Users/ACER/Desktop/random.mp4', w_width=1920,
                             w_height=1080, device=device, conf_thres=float(self.rel) / 100,
                             hide_labels=hide_labels, mod=int(mod_set), set_alarm=int(alarm), weights='weights/custom-v6.pt')


    # lcd 설정
    def lcd_update(self):
        self.ui.lcd_mask_num.display(hanium_detect.detected_mask_count[0])
        self.ui.lcd_sani_num.display(hanium_detect.detected_sani_count[0])
        self.ui.lcd_temp_num.display(hanium_detect.detected_temp_count[0])
        self.ui.lcd_qr_num.display(hanium_detect.detected_qr_count[0])
        self.ui.lcd_total_num.display(
            self.ui.lcd_mask_num.value() + self.ui.lcd_qr_num.value() +
            self.ui.lcd_temp_num.value() + self.ui.lcd_sani_num.value()
        )

    # 시작 버튼 누를 시 동작: 로그 업데이트
    def log_update(self):
        if len(hanium_detect.log_data) > self.index:
            log = f"{hanium_detect.log_data[self.index]['time']}경 {hanium_detect.log_data[self.index]['act']}한 사용자 탐지\n"
            self.ui.text_log.append(log)
            self.index += 1

    # 로그 초기화 버튼 누를 시 동작: 로그 초기화
    def log_clear(self):
        self.ui.text_log.setText("")

    # DB 업데이트 버튼이나 DB 검색의 원래대로 버튼 누를 시 동작: 현 DB 내용으로 업데이트
    def db_update(self):
        self.ui.table_db.setRowCount(0)
        corona_db = pymysql.connect(
            user='root',
            passwd='coldplay96!',
            host='localhost',
            db='corona',
            charset='utf8mb4'
        )
        cursor = corona_db.cursor(pymysql.cursors.DictCursor)
        db_init = """SELECT * FROM log"""
        cursor.execute(db_init)
        db_init_data = cursor.fetchall()
        db_sani_num = 0
        db_temp_num = 0
        db_mask_num = 0
        db_qr_num = 0
        for data in db_init_data:
            index = 0
            self.ui.table_db.insertRow(index)
            time_data = data['time'].strftime('%Y-%m-%d %H:%M:%S')
            self.ui.table_db.setItem(index, 0, QTableWidgetItem(time_data))
            self.ui.table_db.setItem(index, 1, QTableWidgetItem(data['check_act']))
            index += 1
            if data['check_act'] == '손소독':
                db_sani_num += 1
            elif data['check_act'] == '온도계':
                db_temp_num += 1
            elif data['check_act'] == 'QR':
                db_qr_num += 1
            else:
                db_mask_num += 1

        db_total_num = db_sani_num + db_temp_num + db_qr_num + db_mask_num
        self.ui.lcd_db_sani.display(db_sani_num)
        self.ui.lcd_db_temp.display(db_temp_num)
        self.ui.lcd_db_qr.display(db_qr_num)
        self.ui.lcd_db_mask.display(db_mask_num)
        self.ui.lcd_db_total.display(db_total_num)
        self.cursor = cursor

    # DB 초기화 버튼 누를 시 동작: DB 초기화
    def db_clear(self):
        msg = QMessageBox()
        reply = msg.warning(self, '경고!', "DB의 모든 데이터가 지워집니다!\n 그래도 진행하시겠습니까?",
                            QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            table_data_delete = """TRUNCATE log"""
            self.cursor.execute(table_data_delete)
            self.ui.table_db.setRowCount(0)
            self.ui.lcd_db_sani.display(0)
            self.ui.lcd_db_temp.display(0)
            self.ui.lcd_db_qr.display(0)
            self.ui.lcd_db_mask.display(0)
            self.ui.lcd_db_total.display(0)
        else:
            return

    # DB 검색의 검색 버튼 누를 시 동작: 검색 내용에 맞춰 DB 테이블 변경
    def db_search(self):
        self.ui.table_db.setRowCount(0)
        search_data_time = self.ui.time_db.dateTime().toPython()
        search_data_act = self.ui.act_db.currentText()

        db_search_sql = """SELECT * FROM log WHERE log.check_act = %s"""
        self.cursor.execute(db_search_sql, search_data_act)
        db_data = self.cursor.fetchall()

        for data in db_data:
            time = search_data_time - data['time']
            if int(time.total_seconds()) < 0:
                index = 0
                self.ui.table_db.insertRow(index)
                time_data = data['time'].strftime('%Y-%m-%d %H:%M:%S')
                self.ui.table_db.setItem(index, 0, QTableWidgetItem(time_data))
                self.ui.table_db.setItem(index, 1, QTableWidgetItem(data['check_act']))
                index += 1


# Main
if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()

    apply_stylesheet(app, theme='dark_teal.xml')

    sys.exit(app.exec())
