import rawpy
import imageio
import numpy as np
import cv2
import os


def stackImagesKeypointMatching(file_list):
    orb = cv2.ORB_create()

    # disable OpenCL to because of bug in ORB in OpenCV 3.1
    #cv2.ocl.setUseOpenCL(False)

    stacked_image = None
    first_image = None
    first_kp = None
    first_des = None
    for file in file_list:
        image = file
        imageF = file.astype(np.float32) / 255

        # compute the descriptors with ORB
        kp = orb.detect(image, None)
        kp, des = orb.compute(image, kp)

        # create BFMatcher object
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        if first_image is None:
            # Save keypoints for first image
            stacked_image = imageF
            first_image = image
            first_kp = kp
            first_des = des
        else:
             # Find matches and sort them in the order of their distance
            matches = matcher.match(first_des, des)
            matches = sorted(matches, key=lambda x: x.distance)

            src_pts = np.float32(
                [first_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32(
                [kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate perspective transformation
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            w, h, _ = imageF.shape
            imageF = cv2.warpPerspective(imageF, M, (h, w))
            stacked_image += imageF
            print("processing...")

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    cv2.imwrite("stack.png", stacked_image)
    imageio.imsave('stack.tiff', stacked_image)
    #return stacked_image




def raw_loader():
    files = os.listdir(r"C:\Users\darkj\PycharmProjects\Nightium\rawtest")
    raw_array = []

    for i in files:
        with rawpy.imread("rawtest/" + i) as raw:
            print("Loading:" + str(i))
            rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT)
            raw_array.append(rgb)
    return raw_array

def rawpy_test():
    path = 'rawtest/DSC09429.ARW'
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT)
    imageio.imsave("DHT" + 'default.tiff', rgb)
