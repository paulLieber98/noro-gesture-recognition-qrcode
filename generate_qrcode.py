#using: https://www.geeksforgeeks.org/python/generate-qr-code-using-qrcode-in-python/
import qrcode

url = "https://www.noro.co/" #replace with whatever the actual link is for the qrcode-Noro-screen remote

qr = qrcode.make(url) #generates qrcode
qr.save('qrcode.png')

