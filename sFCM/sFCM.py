import numpy as np

class FCM:
    MAX_ITER = 50
    def __init__(self, m, K, imshape):
        '''
        Initializes a sFCM algorithm with parameters

            m       - integer that controls the fuzziness
            K       - number of classes
            imshape - np array (h, w, c)
        '''
        self.m       = m
        self.K       = K
        self.imshape = imshape

        #initialize random memberships
        self.u = self._u_init(*imshape[:2])

        #intialize mapping functions
        self.dec = self.decoder(imshape[1])
        self.enc = self.encoder(imshape[1])


    def decoder(self, w):
        return lambda j : (j // w, j % w)

    def encoder(self, w):
        return lambda i, j : w*i + j

    def _u_init(self, w, h):
        '''
        initialize membership function randomly
        '''

        arr = np.random.random((w*h, self.K))

        #normalize
        arr = arr/arr.sum(axis=1)[:, np.newaxis]

        return arr

    def _vi(self, u, im):
        '''
        Calculate cluster centers
        '''

        vis = []

        for i in range(self.K):
            num = sum([u[j][i]**self.m * im[self.dec(j)] for j in range(np.prod(im.shape[:2]))])

            denom = sum([u[j][i]**self.m for j in range(np.prod(im.shape[:2]))])

            vis.append(num/denom)

        return vis
    
    def get_u(self, im, vis):
        up = np.zeros_like(self.u)

        for j in range(up.shape[0]):
            for i in range(self.K):
                up[j][i] = 1.0/sum([np.linalg.norm(im[self.dec(j)] - vis[i])/np.linalg.norm(im[self.dec(j)] - vis[k]) for k in range(self.K)])**(2.0/(self.m-1))
        return up

            
    def get_class(self, i, j):
        return np.argmax(self.u[self.enc(i,j),:])

    def step(self, im, eps):
        '''
        Runs a single clustering step

        returns a tuple (statistics, converged)
        '''

        #TODO implement convergence criteria based on eps

        vis = self._vi(self.u, im)
        self.u = self.get_u(im, vis)

        stats = np.zeros(im.shape)
        for i in range(stats.shape[0]):
            for j in range(stats.shape[1]):
                if len(self.imshape) == 3:
                    stats[i,j,:] = vis[self.get_class(i,j)]
                elif len(self.imshape) == 2:
                    stats[i,j] = vis[self.get_class(i,j)]

        return stats, False
        
    def run(self, im, eps):
        '''
        Runs the iterative process of clustering

        im  - np array (h, w, c)
        eps - convergence criteria

        Yields statistics for each iteration
        '''

        for c in range(FCM.MAX_ITER):
            stats, finish = self.step(im, eps)

            yield stats, finish
        


    
class sFCM(FCM):
    MAX_ITER = 50
    def __init__(self, m, K, p, q, w_siz, imshape):
        '''
        Initializes a sFCM algorithm with parameters

            m       - integer that controls the fuzziness
            K       - number of classes
            p       - integer that controls importance of membership function
            q       - integer that controls importance of spatial function
            w_size  - odd integer controlling the window size for spatial function (w_size x w_size)
            imshape - np array (h, w, c)
            Note (p,q) = (1,0) => regular FCM
        '''
        assert w_siz % 2 != 0

        super(sFCM, self).__init__(m, K, imshape)

        self.p = p
        self.q = q
        self.w_siz = w_siz

        #intialize spatial function
        self.h = self._h(self.u)

        


    def _nb(self, j, mx):

        i, j = self.dec(j)

        imx, jmx = self.dec(mx -1)
        ip, jp = max(0, i - self.w_siz//2), max(0, j - self.w_siz//2)

        for x in range(ip, ip + self.w_siz):
            for xx in range(jp, jp + self.w_siz):
                yield self.enc(min(imx, x), min(jmx, xx)) 
    
    def _h(self, u):
        h = np.zeros((u.shape[0], self.K)) 

        for j in range(u.shape[0]):
            for i in range(self.K):
                h[j][i] = sum([u[k][i] for k in self._nb(j, u.shape[0])])
        return h 
    
    def _update_u(self, vis):
        '''
        update membership function
        '''

        up = np.zeros_like(self.u)

        for j in range(up.shape[0]):
            for i in range(up.shape[1]):
                up[j, i] = (self.u[j,i]**self.p * self.h[j, i]**self.q)/sum([self.u[j, k]**self.p * self.h[j, k]**self.q for k in range(self.K)])
        
        
        return up

    #@overrides(FCM)
    def step(self, im, eps):
        '''
        Runs a single clustering step

        returns a tuple (statistics, converged)
        '''

        #TODO implement convergence criteria based on eps

        vis = self._vi(self.u, im)
        self.u = self.get_u(im, vis)
        self.h = self._h(self.u)
        self.u = self._update_u(vis)

        stats = np.zeros(im.shape)
        for i in range(stats.shape[0]):
            for j in range(stats.shape[1]):
                if len(self.imshape) == 3:
                    stats[i,j,:] = vis[self.get_class(i,j)]
                elif len(self.imshape) == 2:
                    stats[i,j] = vis[self.get_class(i,j)]

        return stats, False

class csFCM(sFCM):
    MAX_ITER = 30
    def __init__(self, m, K, p, q, w_siz, imshape):
        '''
        Initializes a sFCM algorithm with parameters

            m       - integer that controls the fuzziness
            K       - number of classes
            p       - integer that controls importance of membership function
            q       - integer that controls importance of spatial function
            w_size  - odd integer controlling the window size for spatial function (w_size x w_size)
            imshape - np array (h, w, c)
            Note (p,q) = (1,0) => regular FCM
        '''
        assert w_siz % 2 != 0

        super(csFCM, self).__init__(m, K, p, q, w_siz, imshape)

    def get_f(self, im, vis):
        return self._h(self.u)/self.m

    def step(self, im, eps):
        '''
        Runs a single clustering step

        returns a tuple (statistics, converged)
        '''

        #TODO implement convergence criteria based on eps

        vis = self._vi(self.u, im)
        self.h = self.get_u(im, vis)
        self.cu = self.get_f(im, vis) * self.h
        self.u = self._update_u(vis)

        stats = np.zeros(im.shape)
        for i in range(stats.shape[0]):
            for j in range(stats.shape[1]):
                if len(self.imshape) == 3:
                    stats[i,j,:] = vis[self.get_class(i,j)]
                elif len(self.imshape) == 2:
                    stats[i,j] = vis[self.get_class(i,j)]

        return stats, False

if __name__ == "__main__":
        
    import cv2

    test = cv2.resize(cv2.imread('/home/kristmundur/Documents/KTH/Project Course/PCiDS/sFCM/74.jpeg'), (100,100))
    #test = cv2.cvtColor(cv2.resize(test, (50, 50)), cv2.COLOR_BGR2GRAY)

    #fcm = sFCM(2, 5, 1, 0.5, 3, test.shape)
    fcm = FCM(2, 10, test.shape)
    #fcm = csFCM(3, 2, 1, 0.5, 3, test.shape)

    for c, (stat, finish) in enumerate(fcm.run(test, 0)):
        img = cv2.resize(stat, (334, 334)).astype(np.uint8)
        #cv2.imwrite('gifs/{}.jpeg'.format(c), img)
        cv2.imshow("test {}".format(c), img)
        cv2.waitKey()
    #print(test.shape)
    #fcm.sfcm(test, None)






