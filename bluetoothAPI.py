import bluetooth
import time
from subprocess import call
#import subprocess



class bluetoothAPI:
    def __init__(self):
        self.server_sock = None
        self.client_sock = None
        self.client_address = None
        self.recvdata = None
        self.port = 1
        self.connection_status = False
        self.peers = {}
        self.listed_devs = []
        self.peers = {}
        self.sources = {}
        self.addresses = {}
        self.is_recv = True
        self.alert = False
        self.msg_send = "alert"
    def connect(self):
        self.client_sock,info = self.server_sock.accept()
        self.client_address, psm = info
        
        self.add_text("\naccepted connection from " + str(address))
        self.peers[address] = sock
        self.addresses[sock] = address
        
        
    def start_server(self):
        self.server_sock = bluetooth.BluetoothSocket( bluetooth.RFCOMM )
        self.server_sock.bind(("",self.port))
        self.server_sock.listen(1)
        #self.connect()
        self.is_recv = True
        self.client_sock,self.address = self.server_sock.accept()
        print("Accepted connection from ",self.address)
        #self.connection_status = True
        #return self.client_sock,self.address
    def send(self,data):
        if not data:
            print("no data")
            return
        #str_re = str(data)
        #con = self.check_conection()
        if self.check_conection() and not self.is_recv:
            self.client_sock.send("recv..%s\n"%data)
        
    def received(self):
        try:
            recvdata = self.client_sock.recv(1024)
        except:
            print("connection loss...")
            recvdata = None
            self.connection_status = False
        return recvdata
    def close_connection(self):
        self.client_sock.close()
        self.server_sock.close()
    def check_conection(self):
        try:
            peer_n = self.client_sock.getpeername()
            #print(peer_n)
            self.connection_status = True
        except:
            self.connection_status = False
        return self.connection_status
"""
if __name__ == "__main__":
    call(['sudo', 'hciconfig', 'hci0', 'piscan'])
    text_connect = "Wait for connection..."
    blz = bluetoothAPI()
    print(text_connect)
    blz.start_server()
    while(1):
        conected = blz.check_conection()
        #print(conected)
        if not conected:        
            print(text_connect)
            blz.start_server()
        if conected and blz.is_recv:
            data = blz.received()
            print("Received \"%s\" through Bluetooth"%data)
            str_re = str(data)
            if("end" in str_re):
                print("Exiting")
                #blz.close_connection()
                blz.alert = True
                blz.is_recv = False
        if blz.alert:
            blz.send("hello/n")
            blz.alert = False
"""           
        
                
