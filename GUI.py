import datetime
import sys
from PySide6.QtCore import QDateTime, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QApplication, QMessageBox
from qt_material import apply_stylesheet

import hanium_detect
from ui import Ui_MainWindow


#  pyside6, qt_material 설치 필요 (requirements에 추가했음)
#  폴더 내의 material.css.template을 파이썬이 설치된 폴더의 Lib\site-packages\qt_material 안에 붙여넣어야 한다.
# TODO 로그 기능 완성 , 기능 추가
class MainWindow(QMainWindow):
    # 코로나 CCTV 프로그램 실행 시 넘겨줄 파라미터값 저장 변수
    processor = "GPU"
    label = "ON"
    rel = "50"
    display = "1280x720"
    alarm = "모두"

    # Checkbox 선택 시 무한 호출 방지용 변수 + 시간 변수
    lock_alarm = False
    lock_corona = False
    datetime = QDateTime.currentDateTime()

    def __init__(self):

        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # 아이콘 설정
        self.setWindowIcon(QIcon("./cctv.png"))

        # 시간 출력, self.tick 호출
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
            if self.ui.alarm_ck_light.isChecked() and self.ui.alarm_ck_siren.isChecked() and self.ui.alarm_ck_msg.isChecked():
                self.ui.alarm_ck_all.setChecked(True)

        self.lock_alarm = False

    # 검사할 코로나 소독 행위 선택 시 동작
    # TODO 딕셔너리로 수정, 설정환경창 업데이트, 파라미터값 업데이트
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
            if self.ui.corona_ck_qr.isChecked() and self.ui.corona_ck_temp.isChecked() and self.ui.corona_ck_sani.isChecked():
                self.ui.corona_ck_all.setChecked(True)

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
            f"화면 해상도: {self.display}\n" +
            f"신뢰도: {self.rel}%"
        )

    # 시작 버튼 누를 시 동작 : 파라미터값 업데이트 + lcd 업데이트 + 코로나 CCTV 실행
    def start(self):
        if self.processor == "GPU":
            device = '0'
        else:
            device = 'cpu'
        if self.label == 'ON':
            hide_labels = False
        else:
            hide_labels = True
        win_size = self.display.split('x')

        lcd_timer = QTimer()
        lcd_timer.setInterval(990)
        lcd_timer.timeout.connect(self.lcd_update)
        lcd_timer.start()

        hanium_detect.detect(source='./data/videos/random.mp4', w_width=int(win_size[0]), w_height=int(win_size[1]),
                             device=device, conf_thres=float(self.rel) / 100, hide_labels=hide_labels)

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


# Main
if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()

    apply_stylesheet(app, theme='dark_teal.xml')

    sys.exit(app.exec())
