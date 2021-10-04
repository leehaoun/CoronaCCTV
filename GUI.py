import datetime
import sys
from PySide6.QtCore import QDateTime, QTimer
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow, QApplication, QMessageBox
from qt_material import apply_stylesheet
from ui import Ui_MainWindow


#  pyside6, qt_material 설치 필요 (requirements에 추가했음)
#  폴더 내의 material.css.template을 파이썬이 설치된 폴더의 Lib\site-packages\qt_material 안에 붙여넣어야 한다.

class MainWindow(QMainWindow):
    processor = "GPU"
    label = "ON"
    rel = "50"
    display = "1280x720"
    alarm = "모두"

    lock_alarm = False
    lock_corona = False
    datetime = QDateTime.currentDateTime()

    def __init__(self):

        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.setWindowIcon(QIcon("./cctv.png"))

        self.timer = QTimer()
        self.timer.setInterval(990)
        self.timer.timeout.connect(self.tick)
        self.timer.start()

        self.ui.lb_now.setText(f"현재시각:")

        self.ui.btn_explain_processor.clicked.connect(self.explain)
        self.ui.btn_explain_display.clicked.connect(self.explain)
        self.ui.btn_explain_alarm.clicked.connect(self.explain)
        self.ui.btn_explain_check.clicked.connect(self.explain)
        self.ui.btn_explain_label.clicked.connect(self.explain)
        self.ui.btn_explain_reliability.clicked.connect(self.explain)

        self.ui.spin_rel.valueChanged.connect(self.sync_rel)
        self.ui.hs_rel.valueChanged.connect(self.sync_rel)

        self.ui.alarm_ck_all.stateChanged.connect(self.alarm_check)
        self.ui.alarm_ck_msg.stateChanged.connect(self.alarm_check)
        self.ui.alarm_ck_siren.stateChanged.connect(self.alarm_check)
        self.ui.alarm_ck_light.stateChanged.connect(self.alarm_check)

        self.ui.corona_ck_all.stateChanged.connect(self.corona_check)
        self.ui.corona_ck_sani.stateChanged.connect(self.corona_check)
        self.ui.corona_ck_temp.stateChanged.connect(self.corona_check)
        self.ui.corona_ck_qr.stateChanged.connect(self.corona_check)

        self.ui.radio_cpu.toggled.connect(self.processor_check)
        self.ui.radio_gpu.toggled.connect(self.processor_check)

        self.ui.radio_on.toggled.connect(self.label_check)
        self.ui.radio_off.toggled.connect(self.label_check)

        self.ui.cb_display.currentIndexChanged.connect(self.display_check)

        self.text_load()

    def tick(self):
        now = datetime.datetime.now()
        df = '%Y년 %#m월 %#d일 %H:%M:%S'
        self.ui.lb_now.setText(f"현재시각 : {now.strftime(df)}")

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

    def sync_rel(self):
        if self.sender() == self.ui.spin_rel:
            self.ui.hs_rel.setValue(self.ui.spin_rel.value())
        elif self.sender() == self.ui.hs_rel:
            self.ui.spin_rel.setValue(self.ui.hs_rel.value())

        self.rel = self.ui.spin_rel.value()
        self.text_load()

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

    def processor_check(self):
        if self.sender().isChecked() is True:
            self.processor = self.sender().text()
            self.text_load()

    def label_check(self):
        if self.sender().isChecked() is True:
            self.label = self.sender().text()
            self.text_load()

    def display_check(self):
        self.display = self.sender().currentText()
        self.text_load()

    def text_load(self):
        self.ui.text_status.setText(
            f"프로세서: {self.processor}\n" +
            f"라벨 표시 유무: {self.label}\n" +
            f"경보: {self.alarm}\n" +
            f"화면 해상도: {self.display}\n" +
            f"신뢰도: {self.rel}%"
        )


if __name__ == "__main__":
    app = QApplication()
    window = MainWindow()
    window.show()

    apply_stylesheet(app, theme='dark_teal.xml')

    sys.exit(app.exec())
