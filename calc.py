def dist(x, y):
    x1, y1 = x
    x2, y2 = y
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5


def std_dev(*args):
    s, n = sum(args), len(args)
    mean = s / n
    return (sum((x - mean) ** 2 for x in args) / n) ** 0.5


def features(pts):
    res = dict()
    res['faceL'] = dist(pts[1], pts[17])
    res['faceW'] = dist(pts[0], pts[2])
    res['forehead'] = dist(pts[7], pts[17])
    res['nose'] = dist(pts[7], pts[8])
    res['N2C'] = dist(pts[8], pts[1])
    res['browD'] = dist(pts[4], pts[5])
    res['eyeL'] = dist(pts[9], pts[10])
    res['eyeR'] = dist(pts[11], pts[12])
    res['lipU'] = dist(pts[13], pts[15])
    res['lipL'] = dist(pts[14], pts[16])
    return res


def errors(pts):
    GR = 1.62
    feats = features(pts)
    err = []
    exp = feats['faceW'] * GR
    err.append((exp - feats['faceL']) / exp)
    err.append(std_dev(feats['forehead'], feats['nose'], feats['N2C']))
    exp = feats['browD']
    err.append((2*exp - feats['eyeL'] - feats['eyeR']) / exp)
    exp = feats['lipU'] * GR
    err.append((exp - feats['lipL']) / exp)
    return 100-sum([abs(i) for i in err])
