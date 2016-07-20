import numpy as np, pylab as pyl, scipy as sci
from scipy import interpolate
from numpy import linalg
import sys


rmi=np.array([-0.4,-0.2,0.0,0.2,0.4,0.6,1.0,2.0])
gmr=np.array([-0.5,-0.19,0.15,0.55,0.97,1.16,1.2,1.26])
wmr=np.array([-0.085,0.,0.05,0.07,0.045,-0.03,-.245,-0.940])
splinermi=interpolate.interp1d(rmi,wmr,kind='cubic')
splinegmr=interpolate.interp1d(gmr,wmr,kind='cubic')

gmri=np.linspace(-0.5,1.26,30)
wmri=splinegmr(gmri)
print wmri
wmr2gmr=interpolate.interp1d(wmr,gmr)


class mob:
    def __init__(self,dat, includeRejects = False):
        self.imIds = []
        self.mags = []
        self.filts = []
        self.ras = []
        self.decs = []
        self.times = []
        self.alphas = []
        self.snrs = []
        self.barys = []
        self.deltas = []
        self.trackID=''
        self.nNights = -1
        self.nDets = -1
        self.timeSpan = -1.0
        self.chiSq = -1.0
        self.dof = -1
        self.rchiSq = -1.0
        self.dof = -1
        self.a = -1.0
        self.e = -1.0
        self.inc = -1.0
        self.q = -1.0
        self.mpcID = None
        self.includeRejects = includeRejects

        self._grabObs(dat)

        self.alpha = [-1, -1]
        self.H = {}
        self.H['w'] = [-1, -1]
        self.H['g'] = [-1, -1]
        self.H['r'] = [-1, -1]
        self.H['i'] = [-1, -1]
        self.H['z'] = [-1, -1]
        self.H['y'] = [-1, -1]

    def _grabObs(self,dat):
        s=dat[0].split()
        self.trackID = s[2]
        self.nNights = int(float(s[6]))
        self.timeSpan = float(s[8])
        self.nDets = int(float(s[10]))
        self.chiSq = float(s[14])
        self.dof = int(float(s[16]))
        self.rchiSq = float(s[18])
        self.a = float(s[20])
        self.e = float(s[22])
        self.inc = float(s[24])
        self.q = float(s[26])

        for ii in range(1,len(dat)):
            s=dat[ii].split()

            fwhm = float(s[11])
            rejFlag = int(float(s[15]))
            if (not self.includeRejects and rejFlag == 1) or fwhm<0.4 or fwhm>2.5: continue

            self.imIds.append(s[0])
            self.times.append(float(s[1]))
            self.ras.append(float(s[3]))
            self.decs.append(float(s[4]))
            self.mags.append(float(s[5]))
            self.snrs.append(float(s[10]))
            self.filts.append(s[6].split('.')[0])
            self.barys.append(float(s[17]))
            self.deltas.append(float(s[18]))
            self.alphas.append(float(s[19]))
        self.mags = np.array(self.mags)
        self.ras = np.array(self.ras)
        self.decs = np.array(self.decs)
        self.times = np.array(self.times)
        self.filts = np.array(self.filts)
        self.alphas = np.array(self.alphas)
        self.snrs = np.array(self.snrs)
        self.dmags = 1.09/self.snrs
        self.barys = np.array(self.barys)
        self.deltas = np.array(self.deltas)

        self.g = None
        self.r = None
        self.i = None
        self.z = None
        self.y = None
        self.w = None
        w=np.where(self.filts == 'g')
        if len(w[0]) >0:
            self.g = [self.times[w], self.mags[w], self.dmags[w], self.alphas[w], self.barys[w], self.deltas[w]]
            self._gw = w
        w=np.where(self.filts == 'r')
        if len(w[0]) >0:
            self.r = [self.times[w], self.mags[w], self.dmags[w], self.alphas[w], self.barys[w], self.deltas[w]]
            self._rw = w
        w=np.where(self.filts == 'i')
        if len(w[0]) >0:
            self.i = [self.times[w], self.mags[w], self.dmags[w], self.alphas[w], self.barys[w], self.deltas[w]]
            self._iw = w
        w=np.where(self.filts == 'z')
        if len(w[0]) >0:
            self.z = [self.times[w], self.mags[w], self.dmags[w], self.alphas[w], self.barys[w], self.deltas[w]]
            self._zw = w
        w=np.where(self.filts == 'y')
        if len(w[0]) >0:
            self.y = [self.times[w], self.mags[w], self.dmags[w], self.alphas[w], self.barys[w], self.deltas[w]]
            self._yw = w
        w = np.where(self.filts == 'w')
        if len(w[0]) > 0:
            self.w = [self.times[w], self.mags[w], self.dmags[w], self.alphas[w], self.barys[w], self.deltas[w]]
            self._ww = w

        #self.fitPhase()
        #if self.alpha <> [-1,-1]: self.displayPhaseCurve()

    def __getitem__(self, item):
        if item == 'w':
            return self.w

        if item == 'g':
            return self.g

        if item == 'r':
            return self.r

        if item == 'i':
            return self.i

        if item == 'z':
            return self.z
        if item == 'y':
            return self.y

    def fitPhase(self, nsim=100, minObsPerFilt=6, minTotalObs=20):

        n=0
        filts=[]
        for i in ['g','r','i','z','y','w']:
            if self[i] <>  None:
                if len(self[i][0]) > minObsPerFilt:
                    filts.append(i)
                    n+=len(self[i][0])

        if minTotalObs>n: return
        Y=np.array([])
        dY=np.array([])
        Alphas=np.array([])

        A=np.zeros((len(filts)+1,n)).astype('float64')
        ind=0
        for k,f in enumerate(filts):
            (t, m, dm, a, b, d) = self[f]
            h=m-5*np.log10(b*d)


            #difft=(t[1:]-t[:-1])*24.
            #diffm=m[1:]-m[:-1]
            #DM=(dm[1:]**2+dm[:-1]**2)**0.5
            #div=abs(diffm/difft)
            #w=np.where((difft<1)&(div>1.8)&(diffm>2*DM))
            #print self.trackID,w,m[w],difft[w]

            Y=np.concatenate([Y,h])
            dY=np.concatenate([dY,dm])
            Alphas=np.concatenate([Alphas,a])
            for j in range(len(m)):
                A[k][ind] = 1.0
                A[len(filts)][ind] = a[j]
                ind+=1

        try:
            AtA = np.dot(A,A.T)
            iAtA = linalg.inv(AtA)
        except:
            return None

        fits = np.zeros((len(filts) + 1, nsim)).astype('float64')
        for ns in range(nsim):
            y=Y+sci.randn(len(Y))*dY
            x=np.dot(iAtA,np.dot(A,y.T))
            fits[:,ns] = np.copy(x)

        bf = np.median(fits,axis=1)
        std = np.std(fits,axis=1)

        self.alpha = [bf[-1], std[-1]]
        for i,f in enumerate(filts):
            self.H[f] = [bf[i], std[i]]
        self.filts = filts


    def displayPhaseCurve(self):
        fig1 = pyl.figure(1)
        pyl.title('{}, alpha={:.2f} +- {:.2f} mag/deg'.format(self.trackID,self.alpha[0],self.alpha[1]))
        sp1 = fig1.add_subplot(111)
        pyl.gca().invert_yaxis()
        #sp2 = fig1.add_subplot(212)
        #pyl.gca().invert_yaxis()
        for i,f in enumerate(self.filts):
            if f=='g': c='g'
            elif f=='r': c='r'
            elif f=='y': c='y'
            elif f=='i': c='b'
            elif f=='z': c='b'
            elif f=='w': c='k'

            (t,m,dm,a,d,b) = self[f]
            h=m-5*np.log10(d*b)
            sp1.errorbar(a,h,yerr=dm,marker='s',linestyle='none',color=c)

            deg=np.linspace(np.min(a),np.max(a),10)
            M=self.H[f][0] + deg*self.alpha[0]

            sp1.plot(deg,M,c+'-')
            #sp2.errorbar(t,h,yerr=dm,c=c)
        print self.H,self.alpha
        pyl.show()



class PSD:
    def __init__(self, fn, includeRejects = False):
        self.objects={}
        self.includeRejects = includeRejects
        self._loadData(fn)

    def _loadData(self,fn):
        with open(fn) as han:
            data=han.readlines()

        inds=[]
        for i in range(len(data)):
            if '##' in data[i]:
                inds.append(i)
        inds.append(len(data))
        self.trackIDs = []
        self.nPhaseFit = 0
        for k in range(len(inds)-1):
            i=inds[k]
            I=inds[k+1]
            ob = mob(data[i:I], self.includeRejects)
            self.objects[ob.trackID] = ob
            self.trackIDs.append(ob.trackID)
            if self.objects[ob.trackID].alpha <> [-1, -1]: self.nPhaseFit +=1
            #print ob.trackID,self.objects[ob.trackID].a

    def __call__(self,name):
        return self.objects[name]

    def __getitem__(self, item):
        tid = self.trackIDs[item]
        return self.objects[tid]

    def __len__(self):
        return len(self.trackIDs)




    ### make a function that returns all observations of a particular filter for ALL objects.
    ### make a function that will work as for i in psd

if __name__=="__main__":
    #psdf = PSD(fn='runA_fraser_v3.out', includeRejects=False)
    psd = PSD(fn='runA_holman_v3.out', includeRejects=False)
    #psdl = PSD(fn='runA_lacerda.out', includeRejects=False)

    gmw=[]
    alpha=[]
    for i in range(len(psd)):
        psd[i].fitPhase()
        if psd[i].H['w']<>[-1, -1] and psd[i].H['g']<>[-1, -1] and psd[i].H['w'][0]>4:
            #psd[i].displayPhaseCurve()
            gmw.append([psd[i].H['g'][0]-psd[i].H['w'][0], (psd[i].H['g'][1]**2 + psd[i].H['w'][1]**2)**0.5])
            alpha.append(psd[i].alpha)
    gmr = np.array(gmw)
    alpha = np.array(alpha)
    pyl.errorbar(gmr[:,0],alpha[:,0],xerr=gmr[:,1],yerr=alpha[:,1],linestyle='none', marker='s')

    #gmr=[]
    #alpha=[]
    #for i in range(len(psd)):
    #    psd[i].fitPhase()
    #    if psd[i].H['r']<>[-1, -1] and psd[i].H['z']<>[-1, -1]:# and psd[i].H['w'][0]>5:
    #        #psd[i].displayPhaseCurve()
    #        gmr.append([psd[i].H['g'][0]-psd[i].H['z'][0], (psd[i].H['g'][1]**2 + psd[i].H['z'][1]**2)**0.5])
    #        alpha.append(psd[i].alpha)
    #gmr = np.array(gmr)
    #alpha = np.array(alpha)
    #pyl.errorbar(gmr[:,0],alpha[:,0],xerr=gmr[:,1],yerr=alpha[:,1],linestyle='none', marker='s')

    pyl.xlabel('(g-w)')
    pyl.ylabel('alpha')
    pyl.show()
