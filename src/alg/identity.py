"""Open-set matching; enrollment appends templates without changing model shape."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


class IdentityMatcher:
    def __init__(self):
        self.scaler = StandardScaler()
        self.projection = PCA(n_components=64, random_state=42)
        self.threshold = 0.

    def fit(self, x, people):
        self.templates = self.projection.fit_transform(self.scaler.fit_transform(x))
        self.labels = np.asarray(people)
        self._index()
        return self

    def _index(self):
        self.index = NearestNeighbors(n_neighbors=1, n_jobs=4).fit(self.templates)

    def transform(self, x):
        return self.projection.transform(self.scaler.transform(x))

    def match(self, x):
        distance, index = self.index.kneighbors(self.transform(x))
        names = self.labels[index[:,0]]
        return names, distance[:,0]

    def predict(self, x):
        names, distances = self.match(x)
        return np.where(distances<=self.threshold, names, '未注册用户'), distances

    def enroll(self, name, x):
        if not name or len(x)<3:
            raise ValueError('注册至少需要 3 帧及非空用户名')
        self.templates = np.concatenate([self.templates,self.transform(x)])
        self.labels = np.concatenate([self.labels,np.repeat(name,len(x))])
        self._index()

    def calibrate(self, known_x, known_y, unknown_x):
        names, genuine = self.match(known_x)
        _, impostor = self.match(unknown_x)
        thresholds = np.unique(np.quantile(np.concatenate([genuine,impostor]), np.linspace(0,1,301)))
        # Choose on calibration data only; prefer an empirical FAR <= 5%.
        feasible = [t for t in thresholds if (impostor<=t).mean()<=.05]
        self.threshold = float(max(feasible) if feasible else np.nextafter(impostor.min(), -np.inf))
        return {'threshold':self.threshold, 'far':float((impostor<=self.threshold).mean()),
                'frr':float((genuine>self.threshold).mean()),
                'closed_set_accuracy':float((names==known_y).mean())}

    def evaluate(self, known_x, known_y, unknown_x):
        names, distances = self.match(known_x)
        _, unknown_d = self.match(unknown_x)
        closed = float((names==known_y).mean())
        far = float((unknown_d<=self.threshold).mean())
        frr = float((distances>self.threshold).mean())
        return {'accuracy':closed,'accepted_correct_rate':float(((names==known_y)&(distances<=self.threshold)).mean()),
                'far':far,'frr':frr,'threshold':self.threshold,'known_samples':len(known_x),
                'unknown_samples':len(unknown_x),'target':.95,
                'passed':closed>.95 and .01<=far<=.05 and .05<=frr<=.10}
