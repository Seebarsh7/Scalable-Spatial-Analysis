# -*- coding: utf-8 -*-
# Source code from: https://github.com/googollee/eviltransform/blob/master/python/eviltransform/__init__.py

import math
import numpy as np
import pandas as pd

np.set_printoptions(precision=16)

# __all__ = ['wgs2gcj', 'gcj2wgs', 'gcj2wgs_exact',
#            'distance', 'gcj2bd', 'bd2gcj', 'wgs2bd', 'bd2wgs']

earthR = 6378137.0

def outOfChina(lat, lng):
    a = np.logical_and(lng <= 137.8347, lng >= 72.004)
    b = np.logical_and(lat <= 55.8271, lat>= 0.8293)
    res = np.logical_not(np.logical_and(a,b))
    return res


def transform(x, y):
	xy = x * y
	absX = np.sqrt(np.absolute(x))
	xPi = x * math.pi
	yPi = y * math.pi
	d = 20.0*np.sin(6.0*xPi) + 20.0*np.sin(2.0*xPi)
    
	lat = d
	lng = d
    
	lat = 20.0*np.sin(yPi) + 40.0*np.sin(yPi/3.0) + lat
	lng = 20.0*np.sin(xPi) + 40.0*np.sin(xPi/3.0) + lng

	lat += 160.0*np.sin(yPi/12.0) + 320*np.sin(yPi/30.0)
	lng += 150.0*np.sin(xPi/12.0) + 300.0*np.sin(xPi/30.0)
    
	lat *= 2.0 / 3.0
	lng *= 2.0 / 3.0
    
	lat += -100.0 + 2.0*x + 3.0*y + 0.2*y*y + 0.1*xy + 0.2*absX
	lng += 300.0 + x + 2.0*y + 0.1*x*x + 0.1*xy + 0.1*absX
	return lat, lng


def delta(lat, lng):
    ee = 0.00669342162296594323
    dLat, dLng = transform(lng-105.0, lat-35.0)
    radLat = lat / 180.0 * math.pi
    magic = np.sin(radLat)
    magic = 1 - ee * magic * magic
    sqrtMagic = np.sqrt(magic)
    dLat = (dLat * 180.0) / ((earthR * (1 - ee)) / (magic * sqrtMagic) * math.pi)
    dLng = (dLng * 180.0) / (earthR / sqrtMagic * np.cos(radLat) * math.pi)
    #print('delta success')
    return dLat, dLng


def wgs2gcj(wgsLat, wgsLng):
    #index = np.where(outOfChina(wgsLat, wgsLng)==True)
    index = np.where(outOfChina(wgsLat, wgsLng)==False)[0]
    dlat, dlng = delta(wgsLat, wgsLng)
    wgsLat[index] = dlat[index] + wgsLat[index]
    wgsLng[index] = dlng[index] + wgsLng[index]
    return wgsLat, wgsLng
    

def gcj2wgs(gcjLat, gcjLng):
    if outOfChina(gcjLat, gcjLng):
        return gcjLat, gcjLng
    else:
        dlat, dlng = delta(gcjLat, gcjLng)
        return gcjLat - dlat, gcjLng - dlng


def gcj2wgs_exact(gcjLat, gcjLng):
    shape = gcjLat.shape[0]
    initDelta = [0.01] * shape
    threshold = 0.000001
    dLat = dLng = initDelta
    mLat = gcjLat - dLat
    mLng = gcjLng - dLng
    pLat = gcjLat + dLat
    pLng = gcjLng + dLng
    for i in range(30):
        wgsLat = (mLat + pLat) / 2
        wgsLng = (mLng + pLng) / 2
        tmplat, tmplng = wgs2gcj(wgsLat, wgsLng)
        wgsLat = (mLat + pLat) / 2
        wgsLng = (mLng + pLng) / 2
        #print('temp success')
        dLat = tmplat - gcjLat
        dLng = tmplng - gcjLng
        if (np.logical_and(np.absolute(dLat).max() < threshold, np.absolute(dLng).max() < threshold) == True):
            return wgsLat, wgsLng
        pLat[dLat > 0] = wgsLat[dLat > 0]
        mLat[dLat < 0] = wgsLat[dLat < 0]
        pLng[dLng > 0] = wgsLng[dLng > 0]
        mLng[dLng < 0] = wgsLng[dLng < 0]
    return wgsLat, wgsLng


def distance(latA, lngA, latB, lngB):
    pi180 = math.pi / 180
    arcLatA = latA * pi180
    arcLatB = latB * pi180
    x = (math.cos(arcLatA) * math.cos(arcLatB) *
         math.cos((lngA - lngB) * pi180))
    y = math.sin(arcLatA) * math.sin(arcLatB)
    s = x + y
    if s > 1:
        s = 1
    if s < -1:
        s = -1
    alpha = math.acos(s)
    distance = alpha * earthR
    return distance