"""A module docstring that describes the nature of the data set, the web site that describes the data set more fully, and contain relevant references to academic literature.
Logic for downloading the data set from the most official internet distribution location possible.
Logic for unpacking and loading that data set into primitive Python data types, if possible."""

from bs4 import BeautifulSoup
from random import sample
import os
from urllib2 import urlopen
import numpy as np
import random
import tarfile
import tabular as tb
import Image
import ImageOps
import skdata.larray as larray


class IMAGENET():
    def __init__(self, path=None):
        if path is None:
            print "Please download data using one of the fetch methods"
        else:
            self.path = path

    @property
    def meta(self):
        """Loads the meta from file, if it exists.
        If it doesn't exist, this means images have not been downloaded"""
        if self.path is None:
            print "You must first download images using one of the fetch methods"
        if not hasattr(self, '_meta'):
            try:
                tabular_load = tb.io.loadbinary(self.path + '/imagenet_meta.npz')
                # This seems like a flaw with tabular's loadbinary.
                self._meta = tb.tabarray(records=tabular_load[0], dtype=tabular_load[1])
            except IOError:
                print 'Meta not found in path'
        return self._meta

    def get2013_Categories(self):
        """Get list of wnids in 2013 ILSCRV Challenge by scraping the challenge's website"""
        name_list = []
        wnid_list = []
        #Grabbed website to extract wnids for all images
        parser = BeautifulSoup(urlopen("http://www.image-net.org/challenges/LSVRC/2013/browse-synsets"))

        def is_a_2013_category(tag):
            """
            Returns true if the tag is a link to a category in the 2013 challenge
            :type tag: tag
            :rtype : boolean
            :param tag: tag object
            """
            if tag.has_attr('href'):
                if 'wnid' in tag['href']:
                    return True
            else:
                return False

        linkTags = parser.findAll(is_a_2013_category)
        for linkTag in linkTags:
            name_list.append(linkTag.string)
            link = linkTag['href']
            wnid_list.append(link.partition('wnid=')[2])
        return wnid_list

    def get_word_dictionary(self):
        words_text = urlopen("http://www.image-net.org/archive/words.txt").readlines()
        word_dictionary = {}
        for row in words_text:
            word_dictionary[row.split()[0]] = ' '.join(row.split()[1:]).rstrip('\n')
        return word_dictionary

    def get_definition_dictionary(self):
        gloss_text = urlopen("http://www.image-net.org/archive/gloss.txt").readlines()
        definition_dictionary = {}
        for row in gloss_text:
            definition_dictionary[row.split()[0]] = ' '.join(row.split()[1:]).rstrip('\n')
        return definition_dictionary

    def fetch(self, wnids, seed=None, num_per_synset=100, firstonly=False, path=os.getcwd(),
              username='ardila', accesskey='bd662acb4866553500f17babd5992810e0b5a439'):
        """
        Stores a random #num images for synsets specified by wnids from the latest release to path specified
        Since files are stored as tar files online, the entire synset must be downloaded to access random images.

        If 'all' is passed as the num argument, all images are stored.

        If the argument firstonly is set to true, then download times can be reduced by only extracting the first
        few images

        This method overwrites previous fetches: files and metadata are deleted
        """
        if not os.path.exists(path + '/'):
            os.makedirs(path + '/')
        wnids = list(wnids)
        random.seed(seed)
        kept_names = []
        kept_wnid_list = []
        if hasattr(self, '_meta'):
            files_to_remove = np.unique(self.meta['filename'])
            for file_to_remove in files_to_remove:
                try:
                    print path + '/' + file_to_remove
                    os.remove(path + '/' + file_to_remove)
                except OSError:
                    print "metadata is stale, clear cache directory"
        for i, wnid in enumerate(wnids):
            synset_names = []
            url = 'http://www.image-net.org/download/synset?' + \
                  'wnid=' + str(wnid) + \
                  '&username=' + username + \
                  '&accesskey=' + accesskey + \
                  '&release=latest'
            print i
            url_file = urlopen(url)
            tar_file = tarfile.open(fileobj=url_file, mode='r|')
            if firstonly and not (num_per_synset == 'all'):
                keep_idx = xrange(num_per_synset)
                for i in keep_idx:
                    tarinfo = tar_file.next()
                    synset_names.append(tarinfo.name)
                    tar_file.extract(tarinfo, path)
            else:
                for tarinfo in tar_file:
                    synset_names.append(tarinfo.name)
                    tar_file.extract(tarinfo, path)
                if num_per_synset == 'all':
                    keep_idx = range(len(synset_names))
                else:
                    keep_idx = sample(range(len(synset_names)), num_per_synset)
                files_to_remove = frozenset(synset_names) - frozenset([synset_names[idx] for idx in keep_idx])
                for file_to_remove in files_to_remove:
                    os.remove(path + '/' + file_to_remove)
            kept_names.extend([synset_names[idx] for idx in keep_idx])
            kept_wnid_list.extend([wnid] * len(keep_idx))
        meta = tb.tabarray(records=zip(kept_names, kept_wnid_list), names=['filename', 'wnid'])
        self._meta = meta
        self.path = path
        tb.io.savebinary('imagenet_meta.npz', self._meta)

    def fetch_2013_ILSCRV_synsets(self, num_per_synset=100, seed=None,
                                  path=os.getcwd(), firstonly=False):
        """
        Stores a random #num images for the 2013 ILSCRV synsets from the latest release.
        Since files are stored as tar files online, the entire synset must be downloaded to access random images.

        If 'all' is passed as the num argument, all images are stored.

        If the argument firstonly is set to true, then download times can be reduced by only extracting the first
        few images

        Returns a tabular meta object that has a record for each image containing 2 fields
            wnid: the wnid of the image
            filename: the filename of the image
        """
        synsets_not_ready_yet = ['n04399382']  # Somehow, teddy bears are not ready for download as of 6/14/2013
        wnids = self.get2013_Categories()
        wnids = set(wnids) - set(synsets_not_ready_yet)
        self.fetch(wnids, seed=seed, num_per_synset=num_per_synset, path=path, firstonly=firstonly)


    def get_images(self, resize_to=(256, 256), mode='L', dtype='float32',
                   crop=None, mask=None, normalize=True):
        """
        Create a lazily reevaluated array with preprocessing specified by the parameters
        resize_to: Image is resized to the tuple given here (note: not reshaped)
        dtype: The datatype of the image array
        mode: 'RGB' or 'L' sepcifies whether or not to store color images
        mask: Image object which is used to mask the image
        crop: array of [minx, maxx, miny, maxy] crop box applied after resize
        normalize: If true, then the image set to zero mean and unit standard deviation
        """
        full_paths = [self.path + '/' + filename for filename in self.meta['filename']]
        return larray.lmap(ImgLoaderResizer(resize_to=resize_to, dtype=dtype, normalize=normalize,
                                            crop=crop, mask=mask, mode=mode), full_paths)


class ImgLoaderResizer(object):
    """
    Class used to lazily evaluate resizing/other pre-processing and loading image from file in an larray
    """

    def __init__(self,
                 resize_to=None,
                 dtype='float32', # We should think about uint8 for images!#
                 normalize=True,
                 crop=None,
                 mask=None,
                 mode='L'):
        assert len(resize_to) == 2, "Image size must be specified by 2 numbers"
        resize_to = tuple(resize_to)
        self.resize_to = resize_to
        if crop is None:
            crop = (0, 0, self.resize_to[0], self.resize_to[1])
        assert len(crop) == 4, "Crop is specified by left, top, right, and bottom margins"
        crop = tuple(crop)
        l, t, r, b = crop
        assert 0 <= l < r <= resize_to
        assert 0 <= t < b <= resize_to
        self._crop = crop
        assert dtype == 'float32', "Only float 32 is supported"
        self.dtype = dtype
        if mode == 'L':
            self._ndim = 3
            self.shape = tuple([r - l, b - t])
        else:
            self._ndim = 4
            self.shape = tuple([r - l, b - t, 3])
        self.normalize = normalize
        self.mask = mask
        self.mode = mode

    def rval_getattr(self, attr, objs):
        if attr == 'shape' and self.shape is not None:
            return self.shape
        if attr == 'ndim' and self._ndim is not None:
            return self._ndim
        if attr == 'dtype':
            return self.dtype
        raise AttributeError(attr)

    def __call__(self, file_path):
        im = Image.open(file_path)
        if im.mode != self.mode:
            im = im.convert(self.mode)
        if np.all(im.size != self.resize_to):
            new_shape = (int(self.resize_to[0]), int(self.resize_to[1]))
            im = im.resize(new_shape, Image.ANTIALIAS)
        if self.mask is not None:
            mask = self.mask
            tmask = ImageOps.invert(mask.convert('RGBA').split()[-1])
            im = Image.composite(im, mask, tmask).convert(self.mode)
        if self._crop != (0, 0,) + self.resize_to:
            im = im.crop(self._crop)
        l, t, r, b = self._crop
        assert im.size == (r - l, b - t)
        imval = np.asarray(im, self.dtype)
        rval = imval
        if self.normalize:
            rval -= rval.mean()
            rval /= max(rval.std(), 1e-3)
        else:
            rval /= 255.0
        assert rval.shape[:2] == self.resize_to
        return rval

