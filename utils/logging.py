#!/usr/bin/env python
# Created By  : Neale Van Stralen
# version ='1.0'
# ---------------------------------------------------------------------------
"""
Custom logging class to handle statistic, image, and GIF saving for experiments. 

"""
# ---------------------------------------------------------------------------

import tensorflow as tf
import io
from utils.git import SaveCurrentGitState
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
from utils.utils import CreatePath

class ExperimentLogger():
    def __init__(self, LOG_PATH):
        self.LOG_PATH = LOG_PATH

        CreatePath(LOG_PATH)
        CreatePath(LOG_PATH+'/images')
        CreatePath(LOG_PATH+'/tb_logs')

        self.writer = tf.summary.create_file_writer(LOG_PATH+'/tb_logs')

    def LogScalar(self, tag,value,step):
        with self.writer.as_default():
            summary = tf.summary.scalar(tag,value,step=tf.cast(step, tf.int64))
        self.writer.flush()

    def SaveImage(self,fig,name,format="png",**kwargs):
        fig.savefig("{}/images/{}.{}".format(self.LOG_PATH,name,format),**kwargs)

    def LogImage(self,image,name,step):
        with self.writer.as_default():
            summary = tf.summary.image(name, image, step=step)
        self.writer.flush()

    def SaveGIF(self,clip,name,fps=10):
        clip = ImageSequenceClip(clip, fps=fps)
        clip.write_gif("{}/images/{}.gif".format(self.LOG_PATH,name), fps=fps)

    def LogMatplotLib(self,figure,name,step):
        with self.writer.as_default():
            summary = tf.summary.image(name, plot_to_image(figure), step=step)
        self.writer.flush()

    def RecordGitState(self):
        SaveCurrentGitState(self.LOG_PATH)


def plot_to_image(figure):
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  figure.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image
