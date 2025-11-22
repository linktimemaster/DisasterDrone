import socket
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 8080

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind((host,port))
    s.listen()

    while True:
        heatmap = np.zeros((7,7))
        # othermap = [[""] * 7]*7

        othermap = np.empty([7, 7], dtype="S7")

        conn = s.accept()
        while True:
            data = conn[0].recv(1024)
            if (data.decode() == 'EOF'):
                break
            dec_data = data.decode().split()
            name = dec_data[0]
            x = int(dec_data[1])
            y = int(dec_data[2])

            # print(f"[{x}][{y}]: {name}")
            
            if name == "broken":
                heatmap[x][y] = 1
                othermap[x][y] = "broken"
            elif name == "normal":
                heatmap[x][y] = 0
                othermap[x][y] = "normal"

        conn[0].close()
        # print(othermap)
        # print(heatmap)

        plt.colorbar(plt.imshow(heatmap, cmap='hot'))
        plt.show()
