from bs4 import BeautifulSoup
import sys
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
        dem1 = 0
        check =0 
        lan = 0
        mssvHoten, tongDRL, listDiemCacMuc, listDiemTongTieuChi, listTenCacMuc, listTenTieuChi  = self.layDiemVaTen()
        listTatCaDanhSachHoatDong = self.docListTatCaHoatDong()
        tempXuat = []
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
                            # dem1 = int(listTenCacMuc[ia][ib].split()[-1].replace("[", '').replace("]", "")) - int(listDiemCacMuc[ia][ib])
                            # print(dem1)
                            # listCacMucCoTheThamGia.append(str(ia + 1)+ " " + str(ib +1))
                            dem1 = 0
                            for ic in listTatCaDanhSachHoatDong[ia][ib]:
                                # if not ic.isdigit() and ic[0] != "-":
                                dem1 += int(ic.split()[-1])
                                lan +=1
                                if dem1 <= dem:
                                    tempXuat.append(ic)
                                    check = 0
                                else:
                                    if lan ==1:
                                        tempXuat.append(ic)
                                    check = 1
                                    break
                            # print(listTenCacMuc[ia][ib].split()[-1].replace("[", '').replace("]", ""))
                            if check == 1:
                                for id in tempXuat:
                                    print(id.replace(id.split()[-1], "").replace(id.split()[0], ""))
                                tempXuat = []
                                dem1=0
                                # check = 0
                                break
                    print("\n")
DeXuat(sys.argv[1], sys.argv[2])