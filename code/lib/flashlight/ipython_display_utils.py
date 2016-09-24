from pylab import *

import IPython.display

def get_inline_video(path):

    video_tag_template = \
    """
    <video controls autoplay loop>
      <source src="data:video/mp4;base64,{0}" type="video/mp4" />
    </video>
    """

    video        = open(path, "rb").read()
    video_base64 = video.encode("base64")
    video_tag    = video_tag_template.format(video_base64)

    return IPython.display.HTML(video_tag)
