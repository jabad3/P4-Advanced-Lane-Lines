from moviepy.editor import VideoFileClip
from IPython.display import HTML
from img_gen import process_image

# specify input and output destinations
input_video_path = '../test_videos/project_video.mp4'
ouput_video_path = '../output_images/final.mp4'

# split input video into frames, pass in the processing function
# then merge frames and output final results
clips = VideoFileClip(input_video_path)
video_clip = clips.fl_image(process_image)
video_clip.write_videofile(ouput_video_path, audio=False)