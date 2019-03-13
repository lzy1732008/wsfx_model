import xml.dom.minidom as par
import re
import code.util.num as numutil
import os
import jieba
import lxml.etree

def getQW(path):
    tree = lxml.etree.parse(path)
    root = tree.getroot()
    for qw in root:
        return qw

def getRDSS(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'BSSLD':
                    for bssldchild in ajjbqkchild:
                        if bssldchild.tag == 'ZJXX':
                            for zjxxchild in bssldchild:
                                if zjxxchild.tag == 'ZJFZ':
                                    for zjfzchild in zjxxchild:
                                        if zjfzchild.tag == 'RDSS':
                                            content = zjfzchild.attrib['value']
    return content
#指控段落
def getZKDL(path):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'ZKDL':
                    content = ajjbqkchild.attrib['value']
    return content

#从新填充了法条内容的文书里提取法条列表
def getFTList(path):
    ftnamelist = []
    ftnrlist = []
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'YYFLNR':
            for yyflfzchild in qwchild:
                if yyflfzchild.tag == 'FLNRFZ':
                    for flnrfzchild in yyflfzchild:
                        flag = 0
                        if flnrfzchild.tag == 'FLMC':
                            flmc = flnrfzchild.attrib['value']
                            flag += 1
                        if flnrfzchild.tag == 'FLNR':
                            flnr = flnrfzchild.attrib['value']
                            flag += 2
                        if flag == 2 and flmc and flnr and flnr != 'NOT FOUND':
                            ftnamelist.append(flmc)
                            ftnrlist.append(flnr)

    return ftnamelist,ftnrlist

#文书QW下面的节点内容获取,如文首、诉讼情况、案件基本情况、裁判分析过程、判决结果这几个的value

def getQWChildContent(path,childname):
    content = ''
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == childname:
            content += qwchild.attrib['value']

    return content



def getFTfromQW(path):
    ftls = []
    qw = getQW(path)
    for qwchild in qw:
        if qwchild.tag == 'CPFXGC':
            for cpfxgcchild in qwchild:
                if cpfxgcchild.tag == 'CUS_FLFT_FZ_RY':
                    for fz in cpfxgcchild:
                        if fz.tag == 'CUS_FLFT_RY':
                            ftls.append(fz.attrib['value'])
    return ftls




# 获取事实内容
def getSSMatchObject(wspath):
    return getRDSS(wspath) + getZKDL(wspath)


# 获取结论内容
def getJLMatchObject(wspath):
    return getQWChildContent(wspath, 'CPFXGC') + getQWChildContent(wspath, 'PJJG')

#获取交通肇事罪的证据记录列表
def getZJ(wspath):
    zjlist = []
    qw = getQW(wspath)
    for qwchild in qw:
        if qwchild.tag == 'AJJBQK':
            for ajjbqkchild in qwchild:
                if ajjbqkchild.tag == 'BSSLD':
                    for bssldchid in ajjbqkchild:
                        if bssldchid.tag == 'ZJXX':
                            for zjxxchild in bssldchid:
                                if zjxxchild.tag == 'ZJFZ':
                                    for zjfzchild in zjxxchild:
                                        if zjfzchild.tag == 'ZJJL':
                                            zjlist.append(zjfzchild.attrib['value'])
    return zjlist


#获取xml任意路径的value值
def getnodecontent(wspath,xmlpath):
    pathlist = xmlpath.split('/')
    print(pathlist)
    tree = lxml.etree.parse(wspath)
    root = tree.getroot()
    point = root
    index = 0
    while(index < len(pathlist)):
        for child in point:
            if child.tag == pathlist[index]:
                point = child
                index += 1
                break
    valuelist = []
    parent = point.getparent()
    for p in parent:
        if p.tag == pathlist[-1]:
            valuelist.append(p.attrib['value'])

# getnodecontent('../data/testws5b/264751.xml','QW/AJJBQK/BSSLD/ZJXX/ZJFZ/ZJJL')


def formatft(ft):
    ft = ft.replace(' ','')
    if ft.find('中华人民')>-1 and ft.find('共和国')==-1:
        index = ft.find('中华人民')
        ft = ft[0:index]+'中华人民共和国'+ft[index+4:]
    if ft.find('婚姻法')==0:
        ft = '中华人民共和国'+ft
    if ft.find('中国人民共和国婚姻法')==0:
        ftname='中国人民共和国婚姻法'
        ft='中华人民共和国婚姻法'+ft[10:]
    if ft.find('最高院')==0:
        ft = ft[3:]
        ft = '最高人民法院'+ft
    if ft.find('条第') > 0:
        index = ft.index('条第')
        ft = ft[:index+1]
    if ft.find('条')<len(ft) and ft.find('条')>-1:
        index = ft.index('条')
        ft = ft[:index + 1]
    if ft.find('最高人民法院关于民事诉讼证据若干规定')>-1:
        ftname='最高人民法院关于民事诉讼证据若干规定'
        index = ft.find(ftname)+len(ftname)
        ft='最高人民法院关于民事诉讼证据的若干规定'+ft[index:]
    if ft.find('最高人民法院关于人民法院审理离婚案件处理子女抚养问题若干具体意见')>-1:
        ftname='最高人民法院关于人民法院审理离婚案件处理子女抚养问题若干具体意见'
        index = ft.find(ftname)+len(ftname)
        ft='最高人民法院关于人民法院审理离婚案件处理子女抚养问题的若干具体意见'+ft[index:]

    if ft.find('最高人民法院关于人民法院审理离婚案件处理子女抚养问题的若干意见') > -1:
        ftname = '最高人民法院关于人民法院审理离婚案件处理子女抚养问题的若干意见'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于人民法院审理离婚案件处理子女抚养问题的若干具体意见' + ft[index:]

    if ft.find('最高人民法院关于审理离婚案件处理子女抚养问题的若干具体意见') > -1:
        ftname = '最高人民法院关于审理离婚案件处理子女抚养问题的若干具体意见'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于人民法院审理离婚案件处理子女抚养问题的若干具体意见' + ft[index:]

    if ft.find('最高人民法院关于人民法院审理离婚案件如何认定夫妻感情确已破裂若干具体意见')>-1:
        ftname='最高人民法院关于人民法院审理离婚案件如何认定夫妻感情确已破裂若干具体意见'
        index = ft.find(ftname)+len(ftname)
        ft='最高人民法院关于人民法院审理离婚案件如何认定夫妻感情确已破裂的若干具体意见'+ft[index:]

    if ft.find('最高人民法院关于审理离婚案件如何认定夫妻感情确已破裂的若干具体意见')>-1:
        ftname='最高人民法院关于审理离婚案件如何认定夫妻感情确已破裂的若干具体意见'
        index = ft.find(ftname)+len(ftname)
        ft='最高人民法院关于人民法院审理离婚案件如何认定夫妻感情确已破裂的若干具体意见'+ft[index:]

    if ft.find('最高人民法院关于适用中华人民共和国婚姻法若干问题解释') > -1:
        ftname = '最高人民法院关于适用中华人民共和国婚姻法若干问题解释'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于适用中华人民共和国婚姻法若干问题的解释' + ft[index:]

    if ft.find('最高人民法院关于适用中华人民共和国婚姻法的若干问题解释') > -1:
        ftname = '最高人民法院关于适用中华人民共和国婚姻法的若干问题解释'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于适用中华人民共和国婚姻法若干问题的解释' + ft[index:]

    if ft.find('最高人民法院关于人民法院审理未办结婚登记而以夫妻名义同居生活案件若干意见') > -1:
        ftname = '最高人民法院关于人民法院审理未办结婚登记而以夫妻名义同居生活案件若干意见'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于人民法院审理未办结婚登记而以夫妻名义同居生活案件的若干意见' + ft[index:]
    if ft.find('最高人民法院关于人民法院审理未办理结婚登记而以夫妻名义同居生活案件若干意见') > -1:
        ftname = '最高人民法院关于人民法院审理未办理结婚登记而以夫妻名义同居生活案件若干意见'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于人民法院审理未办结婚登记而以夫妻名义同居生活案件的若干意见' + ft[index:]

    if ft.find('最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题解释') > -1:
        ftname = '最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题解释'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释' + ft[index:]

    if ft.find('最高人民法院关于处理自首和立功具体应用法律若干问题解释') > -1:
        ftname = '最高人民法院关于处理自首和立功具体应用法律若干问题解释'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于处理自首和立功具体应用法律若干问题的解释' + ft[index:]

    if ft.find('最高人民法院关于审理人身损害赔偿案件适用法律若干问题解释') > -1:
        ftname = '最高人民法院关于审理人身损害赔偿案件适用法律若干问题解释'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于审理人身损害赔偿案件适用法律若干问题的解释' + ft[index:]

    if ft.find('最高人民法院关于适用中华人民共和国刑事诉讼法解释') > -1:
        ftname = '最高人民法院关于适用中华人民共和国刑事诉讼法解释'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于适用中华人民共和国刑事诉讼法的解释' + ft[index:]

    if ft.find('最高人民法院关于刑事附带民事诉讼范围问题规定') > -1:
        ftname = '最高人民法院关于刑事附带民事诉讼范围问题规定'
        index = ft.find(ftname) + len(ftname)
        ft = '最高人民法院关于刑事附带民事诉讼范围问题的规定' + ft[index:]
    if ft.find('最高人民法院关于审理交通肇事案件具体应用法律若干问题的解释')>-1:
        ftname='最高人民法院关于审理交通肇事案件具体应用法律若干问题的解释'
        index = ft.find(ftname)+len(ftname)
        ft = '最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释'+ft[index:]
    if ft.find('最高人民法院关于审理交通肇事案件应用法律若干问题的解释')>-1:
        ftname='最高人民法院关于审理交通肇事案件应用法律若干问题的解释'
        index = ft.find(ftname)+len(ftname)
        ft = '最高人民法院关于审理交通肇事刑事案件具体应用法律若干问题的解释'+ft[index:]
    if ft.find('最高人民法院关于处理自首和立功若干问题的意见')>-1:
        ftname='最高人民法院关于处理自首和立功若干问题的意见'
        index=ft.find(ftname)+len(ftname)
        ft='最高人民法院关于处理自首和立功若干具体问题的意见'+ft[index:]

    if ft.find('若干问题的解释二')==0:
        ftname='若干问题的解释二'
        index=ft.find(ftname)+len(ftname)
        ft='最高人民法院关于适用中华人民共和国婚姻法若干问题的解释二'+ft[index:]
    if ft.find('关于人民法院审理借贷案件的若干意见')==0:
        ftname='关于人民法院审理借贷案件的若干意见'
        index=ft.find(ftname)+len(ftname)
        ft='最高人民法院关于人民法院审理借贷案件的若干意见'+ft[index:]
    if ft.find('最高人民法院关于适用中华人民共和国担保法若干问题的解释二')==0:
        ftname='最高人民法院关于适用中华人民共和国担保法若干问题的解释二'
        index=ft.find(ftname)+len(ftname)
        ft='最高人民法院关于适用中华人民共和国担保法若干问题的解释'+ft[index:]
    if ft.find('最高人民法院关于审理借贷案件的若干意见')==0:
        ftname='最高人民法院关于审理借贷案件的若干意见'
        index=ft.find(ftname)+len(ftname)
        ft='最高人民法院关于人民法院审理借贷案件的若干意见'+ft[index:]


    #过滤数字
    nums = re.findall(r'\d+', ft)
    if len(nums)>0:
        num = nums[0]
        numint = int(num)
        numstr = numutil.rankis(numint)
        index = ft.index(num)
        ft = ft[0:index]+numstr+ft[index+len(num):]
    return ft

def getFTfromWS(path):
    dom = par.parse(path)
    root = dom.documentElement
    childlist = root.getElementsByTagName('QW')
    qw = childlist[0]
    CPFXGCList = qw.getElementsByTagName('CPFXGC')
    ft_list = []
    if CPFXGCList:
       CPFXGC = CPFXGCList[0]
       CUS_FLFT_FZ_RYList = CPFXGC.getElementsByTagName('CUS_FLFT_FZ_RY')
       if  CUS_FLFT_FZ_RYList:
            CUS_FLFT_RYlist = CUS_FLFT_FZ_RYList[0].getElementsByTagName('CUS_FLFT_RY')
            if CUS_FLFT_RYlist:
                for i in range(len(CUS_FLFT_RYlist)):
                    ft = (CUS_FLFT_RYlist[i].getAttribute('value'))
                    ft = str(ft).replace('《','')
                    ft = str(ft).replace('》', '')
                    ft = str(ft).replace('（', '')
                    ft = str(ft).replace('）', '')
                    ft = str(ft).replace('﹤', '')
                    ft = str(ft).replace('﹥', '')
                    ft = str(ft).replace('〈', '')
                    ft = str(ft).replace('〉', '')
                    ft = str(ft).replace('、', '')
                    ft = formatft(ft)
                    ft_list.append(ft)
    return ft_list

def getAjjbqk(path):
    dom=par.parse(path)
    root=dom.documentElement
    childlist=root.getElementsByTagName('QW')
    qw=childlist[0]
    Ajjbqklist=qw.getElementsByTagName('AJJBQK')
    content=''
    if len(Ajjbqklist)>0:
        ajjbqk=Ajjbqklist[0]
        if ajjbqk:
            content=ajjbqk.getAttribute('value')
    return content

#get allnum wenshu > 20k from path,return name list
def getws_largerthan20kb(wspath,allnum):
    wsnamelist= os.listdir(wspath)
    count=0
    final_wsnamelist=[]
    for i in range(0,len(wsnamelist)):
        wsname = wsnamelist[i]
        filesize = os.path.getsize(wspath+'/'+wsname)/1024
        if filesize>20:
            final_wsnamelist.append(wspath+'/'+wsname)
            count+=1
        if count>allnum:
            break
    return final_wsnamelist

def count(path):
    ft = '中华人民共和国婚姻法第二十一条'
    countnum=0
    list = os.listdir(path)
    length = len(list)
    if length>5000:
        length=5000
        for i in range(0,length):
            print(i)
            ftlist = getFTfromWS(path + '/' + list[i])
            if ft in ftlist:
                countnum += 1
                print("///////////")
        print(countnum)
    else:
        print('error')

def count2(path):
    ft = '中华人民共和国婚姻法第二十一条'
    countnum=0
    list = os.listdir(path)
    filelist = []
    for file in list:
        filelist.append(path+'/'+file)
    path='/home/ftpwenshu/民事一审案件/分家析产纠纷/2015'
    list = os.listdir(path)
    for file in list:
        filelist.append(path+'/'+file)
    path = '/home/ftpwenshu/民事一审案件/分家析产纠纷/2013'
    list = os.listdir(path)
    for file in list:
        filelist.append(path + '/' + file)
    length = len(filelist)

    if length>5000:
        length=5000
        for i in range(0,length):
            print(i)
            ftlist = getFTfromWS(filelist[i])
            if ft in ftlist:
                countnum += 1
                print("///////////")
        print(countnum)
    else:
        print(length)

def countDF(wspath):
    all=[]
    filepathlist = os.listdir(wspath)
    file=0
    for path in filepathlist:
        if file>1000:
            break
        file=file+1
        qwnr = getQW(wspath+'/'+path)
        list = jieba.cut(qwnr)
        nowlist=[]
        for w in list:
            if len(w)>1 and w not in nowlist:
                all.append(w)
                nowlist.append(w)

    temp=[]
    for w in all:
        if w not in temp:
            count=all.count(w)
            if count>200:
                print(w)
            temp.append(w)






# if __name__=='__main__':
#     # path='/home/ftpwenshu/民事一审案件/婚约财产纠纷/2014'
#     # path = '/home/ftpwenshu/diskb/刑事一审案件/交通肇事罪/2014'
#     # count(path)
#     # print(getZKDL(path+'/544529.xml'))
#     # ft='最高人民法院关于审理离婚案件处理子女抚养问题的若干具体意见rrehy'
#     # print(formatft(ft))
#     # print(getQW('/home/ftpwenshu/diskb/民事一审案件/离婚纠纷/2014/3322129.xml'))
#     path='/home/ftpwenshu/diskb/民事一审案件/离婚纠纷/2013/'
#     countDF(path)