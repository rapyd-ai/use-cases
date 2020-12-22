import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML, Image
import cv2

def display_image(filepath):
    display(Image(filename=filepath, width = 500))

def display_text(text):
    display(HTML("<h2>" + text +"</h2>"))

