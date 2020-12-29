import cv2

# Read image
def cv2_read_image(filepath):
    image = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return image

# Resize image
def cv2_resize_image(img, max_width=1600, max_height=1200):
    img_width = img.shape[1]
    img_height = img.shape[0]

    if img_width >= img_height:
        if img_width > max_width:
            scale_target = max_width  
            width = scale_target
            height = int(img_height * scale_target / img_width)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    if img_width < img_height:
        if img_height > max_height:
            scale_target = max_height  
            height = scale_target
            width = int(img_width * scale_target / img_height)
            dim = (width, height)
            img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    return img

# Write image to disk
def cv2_write_image(cv2_img, filepath):
    # print("Writing output file to ", filepath)
    cv2.imwrite(filepath, cv2_img)

def cv2_read_video(filepath):
    # return cv2_video, length of the video and fps of the video
    video = cv2.VideoCapture(filepath)
    video_length = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = video.get(cv2.CAP_PROP_FPS)

    return video, video_length, video_fps