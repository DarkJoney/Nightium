# Nightium by Dmytro Hlotenko / Powered by Python, OpenCL and libraw
from engine.core import rawpy_test, raw_loader, stackImagesKeypointMatching, stackProcessor


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    #rawpy_test()
    #stackImagesECC(raw_loader())
    stackImagesKeypointMatching(raw_loader(True))
    #stackProcessor(raw_loader())
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
