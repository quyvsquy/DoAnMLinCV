from bs4 import BeautifulSoup
import sys
from datetime import datetime
class DeXuat():
    def __init__(self, outFile, fileData):
        self.outFile = outFile
        self.fileData = fileData
    def docHtmlLayDuoc(self):
        f = open(self.outFile, 'r')
        soup = BeautifulSoup(f.read(), 'html.parser')
        return soup
    def layDiemVaTen(self):
        soup = self.docHtmlLayDuoc()
        mssvHoten = soup.find_all('p')[0].text.replace("Họ", " \nHọ")
        tongDRL = soup.find_all('p')[-1].text

        listDiemTongTieuChi = []
        listDiemCacMuc = []
        listDiemCacMuc.append([])

        listTenCacMuc = []
        listTenCacMuc.append([])
        listTenTieuChi = []

        #Lấy điểm tiêu chí và từng mục
        dem = 0
        for ia in soup.find_all('p'):
            if(ia.text.find("tiêu chí") != -1):
                dem +=1
                listDiemTongTieuChi.append(ia.text.split()[-1])
                listDiemCacMuc.append([])
                # listDiemTongTieuChi.append(ia.text.split()[-1])
                # print(ia.text)
            elif ia.text.find("MSSV") == -1:
                listDiemCacMuc[dem].append(ia.text.split()[-1])
            # print(ia.text)

        #Lấy tên tiêu chí và tên từng mục
        dem = 0
        for ia in soup.find_all("span"):
            if ia.text != "" and ia.text != "Ẩn" and ia.text != "Ẩn Thông tin điểm rèn luyện":
                if ia.text.replace("Ẩn ", "").find( str(dem+1) + ". ") == 0:
                    listTenTieuChi.append(ia.text.replace("Ẩn ", ""))
                    dem +=1
                    listTenCacMuc.append([])
                else:
                    listTenCacMuc[dem-1].append(ia.text.replace("Ẩn ", ""))

        return mssvHoten, tongDRL, listDiemCacMuc, listDiemTongTieuChi, listTenCacMuc, listTenTieuChi

    def docListTatCaHoatDong(self):
        data = open(self.fileData, "r")
        data = data.read().splitlines()
        listTemp = [[] for ia in range(5)] #Tao list co 5 muc
        viTri2 = 0
        listTempVT2 = []
        giaTriTruocCuaVT2 = int(data[0].split()[0].split(".")[1])-1
        for ia in data:
            viTri2 = int(ia.split()[0].split(".")[1])-1
            if giaTriTruocCuaVT2 == viTri2:
                listTempVT2.append(ia)
                if ia == data[-1]:
                    listTemp[int(ia.split()[0].split(".")[0])-1].append(listTempVT2)
            else:
                giaTriTruocCuaVT2 = viTri2
                listTemp[int(listTempVT2[0].split()[0].split(".")[0])-1].append(listTempVT2)
                listTempVT2 = []
                listTempVT2.append(ia)
        return listTemp

    def xuLy(self):
        #Xử  lý
        dem = 0
        mssvHoten, tongDRL, listDiemCacMuc, listDiemTongTieuChi, listTenCacMuc, listTenTieuChi  = self.layDiemVaTen()
        listTatCaDanhSachHoatDong = self.docListTatCaHoatDong()
        listCacMucCoTheThamGia = []
        print(mssvHoten)
        print(tongDRL)
        if int(tongDRL.split()[-1]) >= 100: 
            print("Chúc mừng bạn đã đạt điểm tối đa")
        else:
            for ia in range(5):
                listCacMucCoTheThamGia = []
                if int(listTenTieuChi[ia].split()[-1].replace("[", '').replace("]", "")) > int(listDiemTongTieuChi[ia]):
                    dem =int(listTenTieuChi[ia].split()[-1].replace("[", '').replace("]", "")) - int(listDiemTongTieuChi[ia])
                    print("Mục \'" +listTenTieuChi[ia] + "' bị thiếu: " + str(dem) + " điểm")
                    print("Bạn có thể tham gia các hoạt động: ")
                    for ib in range(len(listDiemCacMuc[ia])):
                        if int(listTenCacMuc[ia][ib].split()[-1].replace("[", '').replace("]", "")) > int(listDiemCacMuc[ia][ib]):
                            listCacMucCoTheThamGia += listTatCaDanhSachHoatDong[ia][ib]
                    if ia == 4:
                        if len(listCacMucCoTheThamGia) == 0:
                            for ib in listTatCaDanhSachHoatDong[ia]:
                                listCacMucCoTheThamGia += ib
                    tempList1 = [] #hoat dong co tgian
                    tempList2 = [] #hoat dong khong tgian
                    for id in listCacMucCoTheThamGia:
                        if id.split()[-1] != "[]" and id.split()[-1] != "[ ]":
                            if not (id.split()[-1].isdigit()  and int(id.split()[-1]) < 12 and id.split()[-2].isdigit()  and int(id.split()[-2]) < 12):
                                tempList2.append(id)
                            else:
                                tempList1.append(id)
                    #Hoat dong theo thoi gian
                    danhSachThang = []
                    for id in tempList1:
                        danhSachThang.append(int(id.split()[-1]))
                    # print(danhSachThang)

                    month =  datetime.now().month
                    # print(danhSachThang)
                    tempInt = self.timHoatDongTrongThangPhuHop(dem, month, danhSachThang, tempList1)
                    if tempInt == -1 or tempInt == -2:
                        for id in range(1,12):
                            if id + month > 12: 
                                tempInt1 = self.timHoatDongTrongThangPhuHop(dem, id + month - 12, danhSachThang, tempList1)
                                if tempInt1  != -1 and tempInt1 != -2:
                                    break
                            else:
                                tempInt2 = self.timHoatDongTrongThangPhuHop(dem, id + month , danhSachThang, tempList1)
                                if tempInt2 != -1 and tempInt2 != -2:
                                    break

                     #Hoat dong k tgian
                    print("Hoặc bạn có thể gia các hoạt động sau: ")
                    self.timHoatDongTheoNam(dem,tempList2)
                    print("\n")
    def timHoatDongTheoNam(self, diem, tempList2):
        dem1 = 0
        tempListHoatDong = []
        lan = 0
        check = 0

        for ic in tempList2:
            if int(ic.split()[-1])  > 0:
                # print(ic)
                dem1 += int(ic.split()[-1])
                lan +=1
                if dem1 <= diem:
                    tempListHoatDong.append(ic)
                    check = 0
                else:
                    if lan ==1:
                        tempListHoatDong.append(ic)
                    dem2 = 0
                    for ia in tempListHoatDong:
                        dem2 += int(ia.split()[-1])
                    if dem2 < diem:
                        tempListHoatDong.append(ic)
                    check = 1
                    break
        if check == 1:
            for id in tempListHoatDong:
                print(id)
        else:
            for id in tempListHoatDong:
                print(id)
            return -2
    def timHoatDongTrongThangPhuHop(self, diem, thang, danhSachThang, tempList1):
        if thang not in danhSachThang:
            return -1
        else:
            print("Bạn có thể tham gia các hoạt động sau vào tháng", str(thang) + ":")
            tempListHoatDong = []
            for ia, ib in enumerate(danhSachThang):
                if ib == thang:
                    # print(tempList1[ia].split()[0])
                    tempString = tempList1[ia][:-1]
                    if tempString[-1].isdigit():
                        tempString = tempString[:-1]
                    tempListHoatDong.append(tempString)
            if self.timHoatDongTheoNam(diem, tempListHoatDong) == -2:
                return -2                



# DeXuat(sys.argv[1], sys.argv[2]).xuLy()