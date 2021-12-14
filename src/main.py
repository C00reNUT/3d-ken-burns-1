#!/usr/bin/env python

import torch
import torchvision
import cupy
import cv2
import getopt
import glob
import h5py
import io
import math
import moviepy
import moviepy.editor
import numpy
import os
import random
import re
import scipy
import scipy.io
import sys
import shutil
import time
import tempfile
import zipfile
import requests
import pydantic

from app.settings import config
from fastapi import FastAPI, FileResponse
from cloudinary.uploader import upload as cloudinary_upload

##########################################################

assert(int(str('').join(torch.__version__.split('.')[0:2])) >= 12) # requires at least pytorch version 1.2.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

objCommon = {}

exec(open('./common.py', 'r').read())
exec(open('./models/disparity-estimation.py', 'r').read())
exec(open('./models/disparity-adjustment.py', 'r').read())
exec(open('./models/disparity-refinement.py', 'r').read())
exec(open('./models/pointcloud-inpainting.py', 'r').read())

##########################################################

app = FastAPI()

cloudinary.config(
	cloud_name=config.CLOUDINARY_CLOUD_NAME,
	api_key=config.CLOUDINARY_API_KEY,
	api_secret=config.CLOUDINARY_API_SECRET,
	secure=True,
)

@app.post("/kbe")
async def autozoom(url: pydantic.HttpUrl):

	# Check the content type of the URL before downloading the content
	try:
		h = requests.head(url.image_url, allow_redirects=True)
	except Exception:
		raise HTTPException(400, detail="Invalid URL for image")
	if "image/jpeg" not in h.headers["Content-Type"] or "image/png" not in h.headers["Content-Type"]:
		raise HTTPException(400, detail="Invalid image file type: expected jpg/jpeg or png")

	# download image and generate input
	img_data = requests.get(url, stream=True).raw
	npyImage = numpy.asarray(bytearray(img_data.read(), dtype='uint8'))
	npyImage = numpy.ascontiguousarray(cv2.imdecode(npyImage, cv2.IMREAD_COLOR))
	intWidth = npyImage.shape[1]
	intHeight = npyImage.shape[0]

	fltRatio = float(intWidth) / float(intHeight)

	intWidth = min(int(1024 * fltRatio), 1024)
	intHeight = min(int(1024 / fltRatio), 1024)

	npyImage = cv2.resize(src=npyImage, dsize=(intWidth, intHeight), fx=0.0, fy=0.0, interpolation=cv2.INTER_AREA)

	# preprocess
	process_load(npyImage, {})

	objFrom = {
		'fltCenterU': intWidth / 2.0,
		'fltCenterV': intHeight / 2.0,
		'intCropWidth': int(math.floor(0.97 * intWidth)),
		'intCropHeight': int(math.floor(0.97 * intHeight))
	}

	objTo = process_autozoom({
		'fltShift': 100.0,
		'fltZoom': 1.25,
		'objFrom': objFrom
	})

	# model infernece
	npyResult = process_kenburns({
		'fltSteps': numpy.linspace(0.0, 1.0, 75).tolist(),
		'objFrom': objFrom,
		'objTo': objTo,
		'boolInpaint': True
	})

	# save output result
    if os.path.isdir(path):
        shutil.rmtree(path)
	os.makedirs(path)

	outputPath = os.join(path, "kenburns.mp4")
	moviepy.editor.ImageSequenceClip(sequence=[ npyFrame[:, :, ::-1] for npyFrame in npyResult + list(reversed(npyResult))[1:] ], fps=25).write_videofile(outputPath)

	# upload movie
	upload_resp = cloudinary_upload(
		outputPath,
		folder="kenburns-outputs",
		resource_type="video",
	)
	return {"output_url": upload_resp["url"]}