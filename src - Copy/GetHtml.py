import argparse
import sys
from random import randint
from time import sleep
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.select import Select
from selenium import webdriver
from bs4 import BeautifulSoup
import GoiY as t
import programDaSua
from selenium.webdriver.firefox.options import Options

# if len(sys.argv) == 4:
#     userName = sys.argv[1]
#     passWord = sys.argv[2]
#     masvCanTra = sys.argv[3]
# else:
#     print("error!")
#     exit()
URL ="https://drl.uit.edu.vn"
URL1 = "https://drl.uit.edu.vn/quantri/xemdiem_sinhvien"

class GetHtml:
    def __init__(self, username, password, mssvCanTra, HocKy):
        self.username = username
        self.password = password
        options = Options()
        options.headless = True
        self.driver = webdriver.Firefox(options=options)
        # self.driver = webdriver.PhantomJS()
        self.mssvCanTra = mssvCanTra
        if self.mssvCanTra == "Unknown":
            print("Unknown")
            self.driver.quit()
            exit()
        self.HocKy = HocKy
    def login(self):
        self.driver.get(URL)
        username = self.driver.find_element_by_css_selector("#edit-name")
        username.send_keys(self.username)
        password = self.driver.find_element_by_css_selector("#edit-pass")
        password.send_keys(self.password)
        btnLogin = self.driver.find_element_by_css_selector("#edit-submit")
        btnLogin.click()
    def TimMSSV(self):
        self.driver.get(URL1)
        edit_masv = self.driver.find_element_by_css_selector("#edit-masv")
        edit_masv.send_keys(self.mssvCanTra)
        btnTim = self.driver.find_element_by_css_selector("#edit-btn-tim")
        btnTim.click()
    def TimMSSVTheoHocKy(self):
        hk = Select(self.driver.find_element_by_css_selector("#edit-xemdiem-bangdiem"))
        hk.select_by_index(self.HocKy)

        # hk = driver.find_element_by_css_selector("#edit-xemdiem-bangdiem > option:nth-child(2)")
        # hk.click()
        btnTim = self.driver.find_element_by_css_selector("#edit-btn-tim")
        btnTim.click()        
    # def ChuyenHky(self):
    #     if self.HocKy == "1":
    def Ghifile(self):
        outHtml = open( self.mssvCanTra + "out.html", "w")
        outHtml.write(self.driver.page_source)
        outHtml.close()
        # self.driver.quit()
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Auto get HTML form drl')
    parser.add_argument('--u', default=None, required=True, help='username')
    parser.add_argument('--p', default=None, required=True, help='password')
    parser.add_argument('--l', default=None, required=False, help='Duong dan cua hinh dau vao')
    # parser.add_argument('--m', default=17520960, required=False, help='mssvCanTra')
    parser.add_argument('--h', default="1", required=False, help='HocKy. Nhap vao so thu tu so voi hoc ky hien tai')

    try:
        options = parser.parse_args()
    except:
        parser.print_help()
        sys.exit(0)
    layMssv = programDaSua.main(options.l)
    drl = GetHtml(options.u, options.p, layMssv, options.h)
    drl.login()
    drl.TimMSSV()
    drl.TimMSSVTheoHocKy()
    drl.Ghifile()
    t.DeXuat(layMssv + "out.html", "data.txt").xuLy()