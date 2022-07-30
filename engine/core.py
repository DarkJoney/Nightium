import rawpy
import imageio
import numpy as np
import cv2
import os
from multiprocessing import Pool, cpu_count
from timeit import default_timer as timer

def stackImagesKeypointMatching(file_list):
    start = timer()
    orb = cv2.ORB_create()
    print("Starting stacking...")
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
    #cv2.imwrite("stack.png", stacked_image)  #it's broken, bad color space or whatever i don't care
    imageio.imsave('stacks.tiff', stacked_image)
    end = timer()
    print(f'elapsed time: {end - start}')

def stackImagesKeypointMatchingMULTI(f1, f2):

    file_list = []
    file_list.append(f1)
    file_list.append(f2)
    orb = cv2.ORB_create()
    print("Starting stacking...")
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
        print("processing...")
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
            #stacked_image += imageF

            return imageF


def stackProcessor(file_list):
    start = timer()
    stack_pairs = []
    first = file_list[0]

    for i in file_list:
        target_pair = []
        target_pair.append(first)
        target_pair.append(i)
        stack_pairs.append(target_pair)
    stack_pairs.pop(0)
    #print(stack_pairs)
    with Pool() as pool:
        res = pool.starmap(stackImagesKeypointMatchingMULTI, stack_pairs)
    stacked_image = res[0]
    res.pop(0)
    for i in res:
        stacked_image = stacked_image + i

    stacked_image /= len(file_list)
    stacked_image = (stacked_image*255).astype(np.uint8)
    #cv2.imwrite("stack.png", stacked_image)  #it's broken, bad color space or whatever i don't care
    imageio.imsave('stack.tiff', stacked_image)
    end = timer()
    print(f'elapsed time: {end - start}')
def raw_processor(i):
    with rawpy.imread("rawtest/" + i) as raw:
        print("Loading:" + str(i))
        rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT, use_camera_wb=True)
        return rgb


def raw_loader():
    files = os.listdir(r"C:\Users\darkj\PycharmProjects\Nightium\rawtest")
    raw_array = []
    with Pool() as pool:
        res = pool.map(raw_processor, files)
    for i in res:
        raw_array.append(i)
    return raw_array


def rawpy_test():
    path = 'rawtest/DSC09429.ARW'
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(demosaic_algorithm=rawpy.DemosaicAlgorithm.DHT, use_camera_wb=True)
    imageio.imsave("DHT" + 'default.tiff', rgb)
