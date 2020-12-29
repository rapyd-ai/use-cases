import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML, Image
import cv2

def display_image(filepath):
    display(Image(filename=filepath, width = 500))

def display_text(text):
    display(HTML("<h2>" + text +"</h2>"))

# Write image to disk
def cv2_write_image(cv2_img, filepath):
    # print("Writing output file to ", filepath)
    cv2.imwrite(filepath, cv2_img)