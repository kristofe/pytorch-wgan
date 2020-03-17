# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
#import tensorflow as tf
import numpy as np
import scipy.misc
from io import BytesIO
from torch.utils.tensorboard import SummaryWriter
import math
import torch
import torchvision



class Logger(object):

    def __init__(self, log_dir):
        """Create a summary writer logging to log_dir."""
        self.writer = SummaryWriter(log_dir)

    def scalar_summary(self, tag, value, step):
        """Log a scalar variable."""
        self.writer.add_scalar(tag, value, step)

    def image_summary(self, tag, images, step, ncols=4):
        """Log a list of images."""
        nrows = math.ceil(len(images))//ncols
        if type(images) == list:
            image_grid = np.stack(images,axis=0)
            image_grid = torch.from_numpy(image_grid)
            image_grid = torchvision.utils.make_grid(image_grid,nrows, padding=1)
        else:
            image_grid = torchvision.utils.make_grid(images,nrows, padding=1)
        image_grid = image_grid * 0.5 + 0.5
        self.writer.add_image(tag, image_grid, step)

    def histo_summary(self, tag, values, step, bins=1000):
        """Log a histogram of the tensor of values."""
        '''
        # Create a histogram using numpy
        counts, bin_edges = np.histogram(values, bins=bins)

        # Fill the fields of the histogram proto
        hist = HistogramProto()
        hist.min = float(np.min(values))
        hist.max = float(np.max(values))
        hist.num = int(np.prod(values.shape))
        hist.sum = float(np.sum(values))
        hist.sum_squares = float(np.sum(values ** 2))

        # Drop the start of the first bin
        bin_edges = bin_edges[1:]

        # Add bin edges and counts
        for edge in bin_edges:
            hist.bucket_limit.append(edge)
        for c in counts:
            hist.bucket.append(c)

        # Create and write Summary
        summary = Summary(value=[Summary.Value(tag=tag, histo=hist)])
        self.writer.add_summary(summary, step)
        self.writer.flush()
        '''