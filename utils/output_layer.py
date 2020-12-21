import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML, Image
import cv2

class OutputLayer:

    def display_image(self, filepath):
        display(Image(filename=filepath, width = 500))

    def display_text(self, text):
        display(HTML("<h2>" + text +"</h2>"))