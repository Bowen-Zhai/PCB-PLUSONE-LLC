from smb.SMBConnection import SMBConnection


def mkdir(dirname):
    main_SharedFile_path = 'test'
    conn.createDirectory(main_SharedFile_path, dirname)
    print("directory has been created")
def uploadfile(filename):
    data = open(filename,'rb')
    # include subfile path
    file = 'test3/' + filename
    main_SharedFile_path = 'test'
    conn.storeFile(main_SharedFile_path,file,data)
    print ("file has been uploaded")


userID = 'nas'
password = 'qwe121'
client_machine_name = 'DESKTOP-xxxxxx'
server_name = 'xxxxx'
server_ip = 'xxx.xxx.x.x'
conn = SMBConnection(userID, password, client_machine_name, server_name)
conn.connect(server_ip)
# create a shared folder
# mkdir("test3")
file = "NAS info.docx"
uploadfile(file)