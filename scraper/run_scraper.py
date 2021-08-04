import time
import itertools
import googlemaps
import urllib
import numpy as np
import pdb
import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import geopandas as gpd
import os
import glob
import matplotlib.pyplot as plt
from shapely.ops import cascaded_union
import numpy as np
import shapely
from shapely.geometry import Point, Polygon
import random
import time
def random_points_in_polygon(number, polygon):
    points = []
    min_x, min_y, max_x, max_y = polygon.bounds
    i= 0
    while i < number:
        point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
        if polygon.contains(point):
            points.append(point)
            i += 1
    return points 
import requests
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


class StreetViewer(object):
    def __init__(self, api_key, location,index, size="256x256",
                 folder_directory='./output/', verbose=True,radius=50):
        """
        This class handles a single API request to the Google Static Street View API
        api_key: obtain it from your Google Cloud Platform console
        location: the address string or a (lat, lng) tuple
        size: returned picture size. maximum is 640*640
        folder_directory: directory to save the returned objects from request
        verbose: whether to print the processing status of the request
        """
        # input params are saved as attributes for later reference
        self._key = api_key
        self.location = location
        self.radius = radius
        self.size = size
        self.folder_directory = folder_directory
        # call parames are saved as internal params
        self._meta_params = dict(key=self._key,
                                location=self.location,
                                radius=self.radius)
        self._pic_params = dict(key=self._key,
                               location=self.location,
                               size=self.size,
                               radius=self.radius)
        self.verbose = verbose
        self.index = str(index)
    
    def get_meta(self):
        """
        Method to query the metadata of the address
        """
        # saving the metadata as json for later usage
        # "/"s are removed to avoid confusion on directory
        # self.meta_path = "{}/meta_{}.json".format(
        #     self.folder_directory, str(self.location).replace("/", ""))
        self._meta_response = requests.get(
            'https://maps.googleapis.com/maps/api/streetview/metadata?',
            params=self._meta_params)
        # turning the contents as meta_info attribute
        self.meta_info = self._meta_response.json()
        # meta_status attribute is used in get_pic method to avoid
        # query when no picture will be available
        self.meta_status = self.meta_info['status']
        # if self._meta_response.ok:
        #     if self.verbose:
        #         print(">>> Obtained Meta from StreetView API:")
        #         print(self.meta_info)
        #     with open(self.meta_path, 'w') as file:
        #         json.dump(self.meta_info, file)
        # else:
        #     print(">>> Failed to obtain Meta from StreetView API!!!")
        self._meta_response.close()
        return self.meta_info
    
    def get_pic(self):
        """
        Method to query the StreetView picture and save to local directory
        """
        meta_fpath = os.path.join(self.folder_directory,"meta",self.index,self.meta_info["pano_id"])
        os.makedirs(meta_fpath,exist_ok=True)
        meta_path = os.path.join(meta_fpath,"meta.json")

        pic_fpath = os.path.join(self.folder_directory,"images",self.index,self.meta_info["pano_id"])
        os.makedirs(pic_fpath,exist_ok=True)

        with open(meta_path, 'w') as file:
                json.dump(self.meta_info, file)
        for heading in [0,90,180,270]:
            self.pic_path = os.path.join(pic_fpath,"pic_{}_.jpg".format(heading))           
            # only when meta_status is OK will the code run to query picture (cost incurred)
            self._pic_params["heading"] = heading
            if self.meta_status == 'OK':
                # if self.verbose:
                #     print(">>> Picture available, requesting now...")
                self._pic_response = requests.get(
                    'https://maps.googleapis.com/maps/api/streetview?',
                    params=self._pic_params)
                self.pic_header = dict(self._pic_response.headers)
                if self._pic_response.ok:
                    # if self.verbose:
                    #     print(f">>> Saving objects to {self.folder_directory}")
                    with open(self.pic_path, 'wb') as file:
                        file.write(self._pic_response.content)
                    self._pic_response.close()
                
            else:
                print(">>> Picture not available in StreetView, ABORTING!")
        if self.verbose:
                    print(">>> COMPLETE!")
            
    def display_pic(self):
        """
        Method to display the downloaded street view picture if available
        """
        if self.meta_status == 'OK':
            plt.figure(figsize=(10, 10))
            img=mpimg.imread(self.pic_path)
            imgplot = plt.imshow(img)
            plt.show()
        else:
            print(">>> Picture not available in StreetView, ABORTING!")



streetview_API_key = '' 


df = gpd.read_file("../input/processed/grid.shp")

NUM_IMAGES = 2000 
for i, row in df.iterrows():

    print(row["index"])
    start_time = time.time()
    os.makedirs("output/images/{}".format(row["index"]),exist_ok=True)
    os.makedirs("output/meta/{}".format(row["index"]),exist_ok=True)
    panids = [t.split("/")[-1] for t in glob.glob("output/images/{}/*".format(row["index"]))]
    total_images = len(panids) 
    images_needed = NUM_IMAGES - total_images
    while images_needed>0:
        
        sampled_point = random_points_in_polygon(1,row["geometry"])[0]
        lat,lng = sampled_point.y,sampled_point.x
        sv = StreetViewer(api_key=streetview_API_key,
                           location="{},{}".format(lat,lng),index=row["index"],folder_directory="output",radius = 10000)
        meta = sv.get_meta() 
        try:
            if (meta["status"]=="OK"):
                if (meta["pano_id"] in (panids))==False:
                    sv.get_pic()     
                    panids = [t.split("/")[-1] for t in glob.glob("output/images/{}/*".format(row["index"]))]
                    total_images = len(panids)
                    images_needed  = NUM_IMAGES - total_images
                    print("images needed: {}".format(images_needed))
                else:
                    print("Pan id exists: {}".format(meta["pano_id"]))       
            else:
                print("not ok")
            if (time.time() - start_time)/60/60 > 1:
                print("TIME UP!!!!!!!!!!!!!!!!!!!!!!!!!")
                break
            pids = glob.glob("output/images/{}/*".format(row["index"]))
            for pid in pids:
                if len(glob.glob(pid+"/*"))==0:
                    os.rmdir(pid)
        except:
            print("Error",meta["pano_id"])
