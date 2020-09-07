import numpy as np

class sFCM:
    MAX_ITER = 50
    def __init__(self, m, K, p, q, w_siz):
        '''
        Initializes a sFCM algorithm with parameters

            m       - integer that controls the fuzziness
            K       - number of classes
            p       - integer that controls importance of membership function
            q       - integer that controls importance of spatial function
            w_size  - odd integer controlling the window size for spatial function (w_size x w_size)
            Note (p,q) = (1,0) => regular FCM
        '''
        assert w_siz % 2 != 0

        self.m = m
        self.K = K
        self.p = p
        self.q = q
        self.w_siz = w_siz

    def decoder(self, w):
        return lambda j : (j // w, j % w)

    def encoder(self, w):
        return lambda i, j : w*i + j

    def __nb(self, j, mx):

        i, j = self.dec(j)

        imx, jmx = self.dec(mx -1)
        ip, jp = max(0, i - self.w_siz//2), max(0, j - self.w_siz//2)

        for x in range(ip, ip + self.w_siz):
            for xx in range(jp, jp + self.w_siz):
                yield self.enc(min(imx, x), min(jmx, xx)) 

    def __u_init(self, w, h):
        '''
        initialize membership function randomly
        '''

        arr = np.random.random((w*h, self.K))

        #normalize
        arr = arr/arr.sum(axis=1)[:, np.newaxis]

        return arr
    
    def __h(self, u):
        h = np.zeros((u.shape[0], self.K)) 

        for j in range(u.shape[0]):
            for i in range(self.K):
                h[j][i] = sum([u[k][i] for k in self.__nb(j, u.shape[0])])
        return h 

    def __vi(self, u, im):
        '''
        Calculate cluster centers
        '''

        vis = []

        for i in range(self.K):
            num = sum([u[j][i]**self.m * im[self.dec(j)] for j in range(np.prod(im.shape[:2]))])

            denom = sum([u[j][i]**self.m for j in range(np.prod(im.shape[:2]))])

            vis.append(num/denom)

        return vis
    
    def get_u(self, im, vis, u):
        up = np.zeros_like(u)

        for j in range(up.shape[0]):
            for i in range(self.K):
                up[j][i] = 1.0/sum([np.linalg.norm(im[self.dec(j)] - vis[i])/np.linalg.norm(im[self.dec(j)] - vis[k]) for k in range(self.K)])**(2.0/(self.m-1))
        return up

    def __update_u(self, u, h, vis):
        '''
        update membership function
        '''

        up = np.zeros_like(u)

        for j in range(up.shape[0]):
            for i in range(up.shape[1]):
                up[j, i] = (u[j,i]**self.p * h[j, i]**self.q)/sum([u[j, k]**self.p * h[j, k]**self.q for k in range(self.K)])
        
        
        return up
    def sfcm(self, im, eps):
        '''
        Runs the iterative process of clustering

        im  - np array (h, w, c)
        eps - convergence criteria
        '''

        self.dec = self.decoder(im.shape[1])
        self.enc = self.encoder(im.shape[1])
        
        #Initialize membership and spatial function
        u = self.__u_init(*im.shape[:2])
        #h = self.__h(u)
        

        c = 0
        while c < sFCM.MAX_ITER:
            vis = self.__vi(u, im)
            #u   = self.__update_u(u, h, vis)
            u = self.get_u(im, vis, u)
            #h = self.__h(u)

            self.u = u

            c +=1

            img = np.zeros(im.shape)

            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    img[i,j,:] = vis[fcm.get_class(i,j)]

            img = cv2.resize(img, (1000, 200)).astype(np.uint8)
            cv2.imwrite('/home/kristmundur/Documents/KTH/Project Course/sFCM/gifs/{0}.jpeg'.format(c), img)
            #cv2.imshow("test {}".format(c), img)
            #cv2.waitKey()

        
    def get_class(self, i, j):
        return np.argmax(self.u[self.enc(i,j),:])


import cv2

test = cv2.imread('test2.png')
test = cv2.resize(test, (100, 20))

fcm = sFCM(2, 10, 1, 0, 5)

print(test.shape)
fcm.sfcm(test, None)






