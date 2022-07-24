''' make obstacles to an occupancy map'''

import numpy as np

class occupancy:

    @staticmethod
    def create_tree_matrix(radius, dx):
        r = int(radius/dx)
        tree = np.zeros((2 * r + 1, 2 * r + 1))
        for i in range(2 * r + 1):
            for j in range(2 * r + 1):
                if (i - r) ** 2 + (j - r)**2 <= r ** 2:
                    tree[i, j] = 1
        return tree
    
    @staticmethod
    def occupancy_map(canvas, x_list, y_list, dx, radius):
        a, b = canvas[0][1] - canvas[0][0], canvas[1][1] - canvas[1][0]
        X, Y = int(a/dx), int(b/dx)
        tree_occupancy = occupancy.create_tree_matrix(radius, dx)
        matrix = np.zeros((Y, X))
        r = int(radius/dx)
        for i in range(len(x_list)):
            x, y = x_list[i], y_list[i]
            x = round((x - canvas[0][0])/dx)
            y = round((y - canvas[1][0])/dx)
            if x > 1400 or y < 0:
                print('x = %s, error!!!'%x)
            x0, x1, y0, y1, row, col = occupancy.determine_pos(X, Y, r, x, y)
            B = tree_occupancy[y0:y1, x0:x1]
            try:
                matrix[row:row+B.shape[0], col:col+B.shape[1]] += B
            # except:
            #     print(row, col)
            #     print(matrix[row:row+B.shape[0], col:col+B.shape[1]].shape, B.shape)
            # try:
            #     matrix[row:row+B.shape[0], col:col+B.shape[1]] += B
            except:
                print('x = %s, y = %s'%(x, y))
                print(row, col, B.shape, x0, x1, y0, y1)
                
        return matrix


    @staticmethod
    def determine_pos(X, Y, radius, x, y):
        # x coordinates
        if x - radius < 0:
            x0 = radius - x
            col = 0
        else:
            x0 = 0
            col = x - radius
        
        if x + radius > X - 1:
            x1 = X - x + radius 
        else:
            x1 = 2 * radius + 1

        # y coordinates
        if y + radius > Y - 1:
            y0 = y + radius - Y + 1
            row = 0 
        else:
            y0 = 0 
            row = Y - 1 - (y + radius)
        
        if y - radius < 0:
            y1 = 2 * radius + 1 + (y - radius)
        else:
            y1 = 2 * radius + 1
        # convert this xy to matrix coordinate

        return x0, x1, y0, y1, row, col
    
    @staticmethod
    def xy2rowcol(x, y, canvas, dx):
        col = round((x - canvas[0][0])/dx)
        row = round((canvas[1][1] - y)/dx - 1)
        return row, col