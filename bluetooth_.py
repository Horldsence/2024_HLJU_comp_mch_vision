# bluetooth low energy scan
import bluetooth

class MyBluetooth:
    def __init__(self) -> None:
        pass
    def FindDevices(self):
        service = bluetooth.discover_devices()
        devices = service.discover(2)

        for address, name in devices.items():
            print("name: {}, address: {}".format(name, address))

    def ReceiveData(self, client_socket):
        data = client_socket.recv(1024)
        return data

    def SendData(self, client_socket, data):
        client_socket.send(data)