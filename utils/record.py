
# Defaults to
try:
    import tensorflow.summary.SummaryWriter as SummaryWriter
    import tf.Summary as Summary
except:
    print("Could not find Tensorflow. Using pytorch backend for logging.")
    import tensorflow.summary.SummaryWriter as SummaryWriter

import json
from utils import CreatePath


class CreateWriter:
    def __init__(self,filePath):
        """Initializes a writer instance which can be used to log training
        statistics and messages.

        Parameters
        ----------
        filePath : str
            Relative or full filepath to save location
        backend : {"Tensorflow","PyTorch"}

        Returns
        -------
        N/A
        """

        self.writer = SummaryWriter(filePath)

    def RecordSummary(self, item, step):
        summary = Summary()
        for key, value in item.items():
            summary.value.add(tag=key, simple_value=value)
        writer.add_summary(summary, step)
        writer.flush()

    def RecordDescription(self,msg):
        summary = Summary()
        meta = tf.SummaryMetadata()
        meta.plugin_data.plugin_name = "text"
        summary.value.add(tag="Description", metadata=meta, tensor=text_tensor)
        writer.add_summary(summary)
        writer.flush()

def SaveJson(filePath,fileName,dictionary):
    CreatePath(MODEL_PATH)
    with open(MODEL_PATH+'/netConfigOverride.json', 'w') as outfile:
        json.dump(netConfigOverride, outfile)
